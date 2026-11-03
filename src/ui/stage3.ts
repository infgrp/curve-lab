/**
 * 3단계 화면: 신경망 학습과 숨겨진 점 (SPEC 3단계 UI).
 *
 * - 뉴런 수 선택(2/8/64) → 학습 애니메이션(requestAnimationFrame 마다 일정 epoch, 프레임이 느리면 줄임)
 * - 학습 중에는 학습 손실만. 끝나면 "숨겨진 점 공개" → 검증 점(다이아몬드)과 검증 MSE, 잡음 하한 σ² 기준선
 * - 오늘 방문자 분포 히스토그램 + 뉴런 수별 평균 (ui/scoreboard.ts, localStorage)
 * - "다시 해 보기": 같은 데이터, 새 초기화 시드
 * 수학은 src/core/mlp.ts, optim.ts, scoreboard.ts 가 한다.
 */
import type { Dataset } from '../core/data.ts';
import { TRUE_FN, mse } from '../core/data.ts';
import { HIDDEN_CHOICES, Mlp, MlpTrainer } from '../core/mlp.ts';
import { Rng } from '../core/rng.ts';
import { type ScoreRecord, betterThanFraction, binCounts, binIndex, meansByH } from '../core/scoreboard.ts';
import { legend, stageHeader } from './chrome.ts';
import { NetView } from './netview.ts';
import { COLORS, Plot } from './plot.ts';
import { ScoreStore } from './scoreboard.ts';
import { T } from './text.ko.ts';

export interface Stage3Result {
  H: number;
  trainMse: number;
  valMse: number;
  seed: number;
}

export interface Stage3Options {
  onNext: (result: Stage3Result | null) => void;
  onSkip?: () => void;
  store?: ScoreStore;
  /** 초기화 시드를 만들 때 쓰는 난수원 (기본: 데이터 시드에서 파생) */
  initSeed?: number;
}

export interface Stage3Handle {
  destroy(): void;
  /** 점검용: 애니메이션 없이 끝까지 학습 */
  trainNow(H?: number): void;
  reveal(): void;
  get state(): State;
  /** 점검용: 은닉 뉴런 k 를 고른다 */
  pick(k: number): void;
  /** 학습된 신경망의 예측 함수 (학습 전이면 null) */
  get predictor(): ((x: number) => number) | null;
}

type State = 'idle' | 'training' | 'trained' | 'revealed';

const EPOCHS = 20000;
const EPOCHS_PER_FRAME_INIT = 120;
const FRAME_BUDGET_MS = 9;
const Y_RANGE: readonly [number, number] = [-2.2, 2.2];
const BOARD_RANGE: readonly [number, number] = [0, 0.4];
const BOARD_BINS = 16;
const t = T.stage3;
const POS_EDGE = '#0b6cff';
const L = T.app.legend;

export function mountStage3(root: HTMLElement, data: Dataset, options: Stage3Options): Stage3Handle {
  const train = data.train;
  const sigma2 = data.sigma * data.sigma;
  const store = options.store ?? new ScoreStore();

  root.innerHTML = `
    <section class="stage stage3">
      ${stageHeader(3, t.title, t.lead, t.guide)}
      <div class="stage-body three">
        <div class="glass panel">
          <div class="panel-head"><h2>${t.plotTitle}</h2>${legend([
            { kind: 'dot', color: COLORS.train, label: L.train },
            { kind: 'diamond', color: COLORS.val, label: '숨겨진 점' },
            { kind: 'line', color: COLORS.train, label: '신경망 곡선' },
            { kind: 'dash', color: '#8794a6', label: '진짜 곡선' },
          ])}</div>
          <canvas class="plot" data-role="fit"></canvas>
          <div class="controls">
            <div class="row">
              <span class="row-name">${t.hiddenLabel}</span>
              <div class="seg" role="radiogroup">
                ${HIDDEN_CHOICES.map((h, i) => `<label><input type="radio" name="hidden" value="${h}" ${i === 1 ? 'checked' : ''} /><span>${h}개</span></label>`).join('')}
              </div>
              <button class="btn" type="button" data-role="train">▶ ${t.train}</button>
              <button class="btn ghost" type="button" data-role="retry" disabled>↻ ${t.retry}</button>
              <button class="btn" type="button" data-role="reveal" disabled>◆ ${t.reveal}</button>
              <label class="badge" style="min-height:48px;gap:10px"><input type="checkbox" data-role="true" disabled style="width:24px;height:24px" /> ${t.showTrue}</label>
            </div>
            <div class="progress"><i data-role="prog"></i></div>
            <p class="status" data-role="status" aria-live="polite"></p>
          </div>
        </div>
        <div class="glass panel">
          <div class="panel-head"><h2>${t.netTitle}</h2></div>
          <canvas class="plot" data-role="net"></canvas>
          <div class="controls">
            ${legend([
              { kind: 'line', color: POS_EDGE, label: t.netPos },
              { kind: 'line', color: COLORS.val, label: t.netNeg },
              { kind: 'line', color: COLORS.path, label: t.netSelected },
            ])}
            <p class="hint" data-role="net-hint">${t.netHint}</p>
          </div>
        </div>
        <div class="glass panel">
          <div class="panel-head"><h2>${t.lossChartTitle}</h2><span class="hint">${t.lossChartHint}</span></div>
          <div class="right-stack">
            <canvas class="plot" data-role="losschart"></canvas>
            <div class="stats three">
              <div class="card stat blue"><span class="k">${t.trainMse}</span><output class="v" data-role="train-mse">–</output></div>
              <div class="card stat red"><span class="k">${t.valMse}</span><output class="v" data-role="val-mse">${t.valMseHidden}</output></div>
              <div class="card stat"><span class="k">${t.noise}</span><output class="v">${sigma2.toFixed(4)}</output></div>
            </div>
            <div class="panel-head"><h2 style="font-size:24px">${t.boardTitle}</h2><span class="hint">${t.boardHint}</span></div>
            <canvas class="plot board" data-role="board"></canvas>
            <p class="hint" data-role="board-means"></p>
          </div>
        </div>
      </div>
      <footer class="stage-foot">
        <p class="bridge" data-role="bridge"></p>
        <div class="row">
          <button class="btn ghost" type="button" data-role="skip">${T.app.skip}</button>
          <button class="btn primary" type="button" data-role="next" disabled>${t.finish} →</button>
        </div>
      </footer>
    </section>`;

  const q = <E extends Element>(role: string): E => {
    const el = root.querySelector<E>(`[data-role="${role}"]`);
    if (!el) throw new Error(`요소 없음: ${role}`);
    return el;
  };
  const fitCanvas = q<HTMLCanvasElement>('fit');
  const lossCanvas = q<HTMLCanvasElement>('losschart');
  const boardCanvas = q<HTMLCanvasElement>('board');
  const netCanvas = q<HTMLCanvasElement>('net');
  const netHint = q<HTMLParagraphElement>('net-hint');
  const trainBtn = q<HTMLButtonElement>('train');
  const retryBtn = q<HTMLButtonElement>('retry');
  const revealBtn = q<HTMLButtonElement>('reveal');
  const trueChk = q<HTMLInputElement>('true');
  const nextBtn = q<HTMLButtonElement>('next');
  const prog = q<HTMLElement>('prog');
  const statusEl = q<HTMLParagraphElement>('status');
  const trainOut = q<HTMLOutputElement>('train-mse');
  const valOut = q<HTMLOutputElement>('val-mse');
  const meansEl = q<HTMLParagraphElement>('board-means');
  const bridgeEl = q<HTMLParagraphElement>('bridge');

  // ── 상태 ──
  let state: State = 'idle';
  let H: number = HIDDEN_CHOICES[1];
  let initSeed = options.initSeed ?? ((Math.imul(data.seed, 2654435761) + 1) >>> 0);
  let net: Mlp | null = null;
  let trainer: MlpTrainer | null = null;
  let raf = 0;
  let epochsPerFrame = EPOCHS_PER_FRAME_INIT;
  let lossLog: { epoch: number; loss: number }[] = [];
  let valMse = NaN;
  let records: ScoreRecord[] = store.today();
  let myRecord: ScoreRecord | null = null;
  let selected: number | null = null; // 눌러서 고른 은닉 뉴런
  let complete = false; // 20000 epoch 을 끝까지 돌렸는가 (중간에 멈춘 결과는 기록하지 않는다)

  const fit = new Plot(fitCanvas, { xRange: [-1.2, 1.2], yRange: Y_RANGE, xLabel: 'x', yLabel: 'y' });
  const lossPlot = new Plot(lossCanvas, { xRange: [0, EPOCHS], yRange: [-4, 0.5], xLabel: '반복', padding: { left: 84, right: 48, bottom: 44 } });
  const board = new Plot(boardCanvas, { xRange: BOARD_RANGE, yRange: [0, 5], xLabel: '검증 손실', padding: { left: 56, bottom: 44 } });
  const netView = new NetView(netCanvas);

  function drawFit(): void {
    fit.clear();
    fit.drawAxes();
    if (trueChk.checked && state === 'revealed') fit.drawCurve(TRUE_FN, { color: '#8794a6', width: 2.5, dash: [8, 6] });
    if (net) fit.drawCurve((x) => net!.predict(x), { color: COLORS.train, width: 4.5 });
    if (state === 'revealed') fit.drawPoints(data.val.x, data.val.y, { shape: 'diamond', color: COLORS.val, radius: 7 });
    fit.drawPoints(train.x, train.y, { shape: 'circle', color: COLORS.train, radius: 9 });
    // 고른 뉴런이 담당하는 꺾은선: v·ReLU(w x + b) 를 금색 점선으로, 꺾이는 점 t = −b/w 에 표식
    if (net && selected !== null && selected < net.H) {
      const j = selected;
      const w = net.W1[j];
      const b = net.b1[j];
      const v = net.W2[j];
      fit.drawCurve((x) => v * Math.max(0, w * x + b), { color: COLORS.path, width: 3.5, dash: [10, 7] });
      if (w !== 0) {
        const tk = -b / w;
        if (tk >= -1.2 && tk <= 1.2) fit.drawMarker(tk, 0, { radius: 10, color: COLORS.path });
      }
    }
  }

  function drawNet(): void {
    netView.draw(net, net ? net.H : chosenH(), selected, { input: 'x', output: 'ŷ', relu: t.netRelu });
    if (net && selected !== null && selected < net.H) {
      const w = net.W1[selected];
      const b = net.b1[selected];
      netHint.textContent = t.netPicked(selected + 1, w !== 0 ? -b / w : NaN, net.W2[selected]);
    } else {
      netHint.textContent = !net ? t.netHint : net.H > 8 ? t.netHintMany : t.netHintTrained;
    }
  }

  function drawLoss(): void {
    lossPlot.clear();
    lossPlot.drawAxes({ yTicks: [-4, -3, -2, -1, 0], format: (v) => (v === 0 ? '1' : v < 0 && v >= -4 ? (10 ** v).toString() : String(v)), xTicks: [0, 5000, 10000, 15000, 20000] });
    if (lossLog.length > 1) {
      lossPlot.drawPath(lossLog.map((p) => ({ x: p.epoch, y: Math.log10(Math.max(1e-4, p.loss)) })), { color: COLORS.train, width: 3 }, false);
    }
    // 잡음 하한 σ² 기준선
    const y = Math.log10(sigma2);
    lossPlot.drawPath([{ x: 0, y }, { x: EPOCHS, y }], { color: COLORS.val, width: 2, dash: [8, 6] }, false);
    lossPlot.drawText('σ²', EPOCHS, y, { align: 'right', baseline: 'bottom', color: COLORS.val, size: 24, dx: -4, dy: -4 });
  }

  function drawBoard(): void {
    board.clear();
    const counts = binCounts(records.map((r) => r.val), BOARD_RANGE[0], BOARD_RANGE[1], BOARD_BINS);
    const maxC = Math.max(1, ...counts);
    board.yRange = [0, maxC * 1.15];
    board.drawAxes({ yTicks: [], xTicks: [0, 0.1, 0.2, 0.3, 0.4], grid: false });
    const w = (BOARD_RANGE[1] - BOARD_RANGE[0]) / BOARD_BINS;
    const mine = myRecord ? binIndex(myRecord.val, BOARD_RANGE[0], BOARD_RANGE[1], BOARD_BINS) : -1;
    board.drawBars(BOARD_RANGE[0], w, counts, { highlight: mine });
    const sx = sigma2;
    board.drawPath([{ x: sx, y: 0 }, { x: sx, y: maxC * 1.15 }], { color: COLORS.val, width: 2, dash: [6, 5] }, false);
    if (!records.length) board.drawText(t.boardEmpty, 0.2, maxC * 0.6, { align: 'center', color: '#566579', size: 24 });
    const means = meansByH(records);
    meansEl.textContent = means.length ? means.map((m) => t.boardMean(m.H, m.mean, m.count)).join(' · ') : '';
  }

  function render(): void {
    drawFit();
    drawLoss();
    drawNet();
    const epoch = trainer?.epoch ?? 0;
    prog.style.width = `${(epoch / EPOCHS) * 100}%`;
    trainOut.value = trainer && Number.isFinite(trainer.lastLoss) ? trainer.lastLoss.toFixed(4) : '–';
    valOut.value = state === 'revealed' ? valMse.toFixed(4) : t.valMseHidden;
    valOut.classList.toggle('bad', state === 'revealed' && valMse > 2 * sigma2);
    trainBtn.textContent = state === 'training' ? `■ ${t.stop}` : `▶ ${t.train}`;
    trainBtn.disabled = state === 'revealed';
    retryBtn.disabled = state === 'idle' || state === 'training';
    revealBtn.disabled = state !== 'trained';
    trueChk.disabled = state !== 'revealed';
    nextBtn.disabled = state !== 'revealed';
    root.querySelectorAll<HTMLInputElement>('input[name="hidden"]').forEach((el) => {
      el.disabled = state === 'training';
    });
  }

  function setStatus(text: string, tone: '' | 'good' | 'bad' = ''): void {
    statusEl.textContent = text;
    statusEl.className = `status ${tone}`.trim();
  }

  // ── 학습 ──
  function chosenH(): number {
    const el = root.querySelector<HTMLInputElement>('input[name="hidden"]:checked');
    return Number(el?.value ?? HIDDEN_CHOICES[1]);
  }

  function stopAnim(): void {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }

  function startTraining(animate: boolean): void {
    stopAnim();
    H = chosenH();
    net = Mlp.init(H, new Rng(initSeed));
    if (selected !== null && selected >= H) selected = null;
    trainer = new MlpTrainer(net, train, 0.01);
    lossLog = [{ epoch: 0, loss: net.loss(train) }];
    valMse = NaN;
    myRecord = null;
    complete = false;
    trueChk.checked = false;
    state = 'training';
    drawBoard();
    setStatus(t.training);
    if (!animate) {
      while (trainer.epoch < EPOCHS) {
        trainer.run(Math.min(500, EPOCHS - trainer.epoch));
        lossLog.push({ epoch: trainer.epoch, loss: trainer.lastLoss });
      }
      finishTraining();
      return;
    }
    const frame = () => {
      if (state !== 'training' || !trainer) return;
      const t0 = performance.now();
      trainer.run(Math.min(epochsPerFrame, EPOCHS - trainer.epoch));
      lossLog.push({ epoch: trainer.epoch, loss: trainer.lastLoss });
      const dt = performance.now() - t0;
      // 프레임 예산에 맞춰 epoch 수를 조절한다 (저사양 PC 에서도 화면이 끊기지 않게)
      if (dt > FRAME_BUDGET_MS) epochsPerFrame = Math.max(20, Math.floor(epochsPerFrame * 0.7));
      else if (dt < FRAME_BUDGET_MS / 3) epochsPerFrame = Math.min(1000, Math.ceil(epochsPerFrame * 1.3));
      render();
      setStatus(`${t.training} ${t.epoch(trainer.epoch, EPOCHS)}`);
      if (trainer.epoch >= EPOCHS) {
        finishTraining();
        return;
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  }

  function finishTraining(): void {
    stopAnim();
    state = 'trained';
    complete = true;
    setStatus(`${t.epoch(EPOCHS, EPOCHS)} · ${t.reveal} 버튼을 눌러 보세요.`);
    render();
  }

  function stopTraining(): void {
    stopAnim();
    if (state === 'training') {
      state = 'trained'; // 중간에 멈춰도 공개는 가능 (학습이 덜 된 상태를 보는 것도 배움)
      setStatus('');
    }
    render();
  }

  function reveal(): void {
    if (state !== 'trained' || !net || !trainer) return;
    const pred = net.predictInto(data.val.x, new Float64Array(data.val.x.length));
    valMse = mse(pred, data.val.y);
    state = 'revealed';
    if (complete) {
      records = store.add({ H, val: valMse, train: trainer.lastLoss });
      myRecord = records[records.length - 1] ?? null;
    } else {
      myRecord = null;
    }
    drawBoard();
    // 해석 문구
    const trainL = trainer.lastLoss;
    let msg = '';
    let tone: '' | 'good' | 'bad' = '';
    if (trainL > 0.5 * valMse && valMse > 3 * sigma2) msg = H <= 2 ? t.deadRelu : t.underfit;
    else if (trainL < 0.25 * sigma2 && valMse > 1.3 * sigma2) {
      msg = t.memorized;
      tone = 'bad';
    } else if (valMse < 2 * sigma2) {
      msg = t.good;
      tone = 'good';
    } else msg = t.underfit;
    const others = complete ? records.slice(0, -1) : records;
    const frac = betterThanFraction(others, valMse);
    // 앞선 기록이 3개는 있어야 백분율이 뜻을 가진다
    const rank = complete && others.length >= 3 && Number.isFinite(frac) ? ` ${t.rank(frac)}` : '';
    setStatus(`${msg}${rank}${complete ? '' : ` ${t.partial}`}`, tone);
    bridgeEl.textContent = t.bridge;
    render();
  }

  trainBtn.addEventListener('click', () => {
    if (state === 'training') stopTraining();
    else startTraining(true);
  });
  retryBtn.addEventListener('click', () => {
    initSeed = (initSeed * 1664525 + 1013904223) >>> 0; // 같은 자료, 새 초기화 시드
    bridgeEl.textContent = '';
    startTraining(true);
  });
  revealBtn.addEventListener('click', reveal);
  trueChk.addEventListener('change', drawFit);
  netCanvas.addEventListener('pointerdown', (ev) => {
    const k = netView.hitTest(ev);
    selected = k === null || k === selected ? null : k;
    drawFit();
    drawNet();
  });
  root.querySelectorAll<HTMLInputElement>('input[name="hidden"]').forEach((el) =>
    el.addEventListener('change', () => {
      if (!el.checked) return;
      stopAnim();
      state = 'idle';
      net = null;
      trainer = null;
      lossLog = [];
      valMse = NaN;
      myRecord = null;
      selected = null;
      setStatus('');
      bridgeEl.textContent = '';
      render();
      drawBoard();
    }),
  );
  nextBtn.addEventListener('click', () => {
    stopAnim();
    options.onNext(state === 'revealed' && trainer ? { H, trainMse: trainer.lastLoss, valMse, seed: initSeed } : null);
  });
  q<HTMLButtonElement>('skip').addEventListener('click', () => {
    stopAnim();
    (options.onSkip ?? (() => options.onNext(null)))();
  });

  fit.onResize(drawFit);
  lossPlot.onResize(drawLoss);
  board.onResize(drawBoard);
  netView.onResize(drawNet);
  render();

  return {
    destroy() {
      stopAnim();
      fit.destroy();
      lossPlot.destroy();
      board.destroy();
      netView.destroy();
      root.innerHTML = '';
    },
    trainNow(h?: number) {
      if (h !== undefined) {
        const el = root.querySelector<HTMLInputElement>(`input[name="hidden"][value="${h}"]`);
        if (el) el.checked = true;
      }
      startTraining(false);
    },
    reveal,
    get state() {
      return state;
    },
    pick(k) {
      selected = Number.isInteger(k) && k >= 0 ? k : null;
      drawFit();
      drawNet();
    },
    get predictor() {
      const n = net;
      return n ? (x: number) => n.predict(x) : null;
    },
  };
}
