/**
 * 1단계 화면: 직선 맞추기 (SPEC 1단계 UI).
 *
 * 왼쪽: 자료 점·직선·잔차 선분과 슬라이더 두 개(a, b), 손실 값.
 * 오른쪽: (a, b) 평면의 손실 등고선(로그 간격)과 현재 위치. 지도를 눌러/끌어 직선을 옮길 수 있다.
 * "기계에게 맡기기": 현재 위치에서 경사하강법을 시작해 경로를 애니메이션으로 그린다.
 * 학습률 선택지 3개(0.1·η_max, 0.6·η_max, 1.1·η_max). 발산해도 앱은 멈추지 않는다.
 * 수학은 모두 src/core/linear.ts 가 한다.
 */
import { type Grid, logLevels, makeGrid } from '../core/contour.ts';
import type { Dataset } from '../core/data.ts';
import {
  ETA_CHOICES,
  type LinearParams,
  linearClosedForm,
  linearEtaMax,
  linearGdStep,
  linearLoss,
} from '../core/linear.ts';
import { COLORS, Plot } from './plot.ts';
import { T } from './text.ko.ts';

export interface Stage1Result {
  a: number;
  b: number;
}

export interface Stage1Options {
  onNext: (result: Stage1Result) => void;
  onSkip?: () => void;
  /** 처음 (a, b). 기본은 (0, 0) */
  initial?: LinearParams;
}

export interface Stage1Handle {
  destroy(): void;
  /** 현재 (a, b) */
  get params(): LinearParams;
  /** 테스트·점검용: 애니메이션 없이 경사하강을 끝까지 돌린다 */
  runMachine(choice: keyof typeof ETA_CHOICES, animate?: boolean): void;
}

const A_RANGE: readonly [number, number] = [-3, 3];
const B_RANGE: readonly [number, number] = [-2, 2];
const Y_RANGE: readonly [number, number] = [-2.2, 2.2];
const MAX_STEPS = 400;
const STEPS_PER_FRAME = 1;
const CONVERGE_TOL = 1e-4;
/** 이보다 멀리 가면 발산으로 본다 (지도 범위의 10배 남짓). 숫자가 읽을 수 없게 커지기 전에 멈춘다. */
const DIVERGE_BOUND = 30;
/** 손실 표시 상한: 이보다 크면 숫자 대신 '너무 큼' */
const LOSS_DISPLAY_MAX = 100;

const t = T.stage1;

export function mountStage1(root: HTMLElement, data: Dataset, options: Stage1Options): Stage1Handle {
  const train = data.train;
  const star = linearClosedForm(train);
  const bestLoss = linearLoss(train, star);
  const etaMax = linearEtaMax(train.x);

  // ── DOM ──
  root.innerHTML = `
    <section class="stage stage1">
      <header class="stage-head">
        <h1>${t.title}</h1>
        <ol class="guide">${t.guide.map((g) => `<li>${g}</li>`).join('')}</ol>
      </header>
      <div class="stage-body">
        <div class="panel">
          <canvas class="plot" data-role="fit"></canvas>
          <div class="controls">
            <label class="slider">
              <span class="slider-name">${t.sliderA} <output data-role="a-out"></output></span>
              <input type="range" data-role="a" min="${A_RANGE[0]}" max="${A_RANGE[1]}" step="0.01" />
            </label>
            <label class="slider">
              <span class="slider-name">${t.sliderB} <output data-role="b-out"></output></span>
              <input type="range" data-role="b" min="${B_RANGE[0]}" max="${B_RANGE[1]}" step="0.01" />
            </label>
            <div class="readout">
              <span class="readout-name">${t.loss}</span>
              <output class="readout-value" data-role="loss"></output>
              <span class="readout-hint">${t.lossHint}</span>
            </div>
          </div>
        </div>
        <div class="panel">
          <h2>${t.contourTitle}</h2>
          <p class="hint">${t.contourGuide}</p>
          <canvas class="plot" data-role="map"></canvas>
          <div class="controls">
            <p class="hint">${t.machineGuide}</p>
            <div class="row">
              <span class="row-name">${t.lrLabel}</span>
              <div class="seg" role="radiogroup" data-role="lr">
                <label><input type="radio" name="lr" value="small" /><span>${t.lr.small}</span></label>
                <label><input type="radio" name="lr" value="good" checked /><span>${t.lr.good}</span></label>
                <label><input type="radio" name="lr" value="tooBig" /><span>${t.lr.tooBig}</span></label>
              </div>
              <button class="btn primary" type="button" data-role="machine">${t.machine}</button>
            </div>
            <p class="status" data-role="status" aria-live="polite"></p>
          </div>
        </div>
      </div>
      <footer class="stage-foot">
        <p class="limit" data-role="limit"></p>
        <div class="row">
          <button class="btn" type="button" data-role="skip">${T.app.skip}</button>
          <button class="btn primary" type="button" data-role="next">${T.app.next}</button>
        </div>
      </footer>
    </section>`;

  const q = <E extends Element>(role: string): E => {
    const el = root.querySelector<E>(`[data-role="${role}"]`);
    if (!el) throw new Error(`요소 없음: ${role}`);
    return el;
  };
  const fitCanvas = q<HTMLCanvasElement>('fit');
  const mapCanvas = q<HTMLCanvasElement>('map');
  const aInput = q<HTMLInputElement>('a');
  const bInput = q<HTMLInputElement>('b');
  const aOut = q<HTMLOutputElement>('a-out');
  const bOut = q<HTMLOutputElement>('b-out');
  const lossOut = q<HTMLOutputElement>('loss');
  const statusEl = q<HTMLParagraphElement>('status');
  const limitEl = q<HTMLParagraphElement>('limit');
  const machineBtn = q<HTMLButtonElement>('machine');

  // ── 상태 ──
  let p: LinearParams = { ...(options.initial ?? { a: 0, b: 0 }) };
  let path: LinearParams[] = [];
  let raf = 0;
  let running = false;
  let currentEta = 0;

  // 등고선 격자는 데이터가 정해지면 한 번만 계산
  const grid: Grid = makeGrid(A_RANGE, B_RANGE, 121, 101, (a, b) => linearLoss(train, { a, b }));
  let maxLoss = 0;
  for (const v of grid.v) maxLoss = Math.max(maxLoss, v);
  const levels = logLevels(bestLoss * 1.03, maxLoss, 14);

  const fit = new Plot(fitCanvas, { xRange: [-1.2, 1.2], yRange: Y_RANGE, xLabel: 'x', yLabel: 'y' });
  const map = new Plot(mapCanvas, { xRange: A_RANGE, yRange: B_RANGE, xLabel: 'a', yLabel: 'b' });

  const line = (x: number) => p.a * x + p.b;

  function drawFit(): void {
    fit.clear();
    fit.drawAxes();
    fit.drawResiduals(train.x, train.y, line);
    fit.drawCurve(line, { color: COLORS.train, width: 4 });
    fit.drawPoints(train.x, train.y, { shape: 'circle', color: COLORS.train, radius: 9 });
  }

  function drawMap(): void {
    map.clear();
    map.drawAxes();
    map.drawContours(grid, levels);
    if (path.length > 1) map.drawPath(path.map((v) => ({ x: v.a, y: v.b })));
    map.drawMarker(p.a, p.b, { radius: 12 });
  }

  function syncInputs(): void {
    aInput.value = String(clamp(p.a, A_RANGE));
    bInput.value = String(clamp(p.b, B_RANGE));
    aOut.value = fmt(p.a);
    bOut.value = fmt(p.b);
    const L = linearLoss(train, p);
    const tooBig = !Number.isFinite(L) || L > LOSS_DISPLAY_MAX;
    lossOut.value = tooBig ? t.lossTooBig : L.toFixed(3);
    lossOut.classList.toggle('bad', tooBig);
  }

  function render(): void {
    syncInputs();
    drawFit();
    drawMap();
    limitEl.textContent = linearLoss(train, p) < bestLoss * 1.15 ? `${t.limit} (${t.bestLoss}: ${bestLoss.toFixed(3)})` : '';
  }

  function setParams(next: LinearParams, clearPath = true): void {
    p = next;
    if (clearPath) path = [];
    render();
  }

  // ── 입력 ──
  const onSlide = () => {
    stopMachine();
    statusEl.textContent = '';
    setParams({ a: Number(aInput.value), b: Number(bInput.value) });
  };
  aInput.addEventListener('input', onSlide);
  bInput.addEventListener('input', onSlide);

  // 지도를 눌러/끌어 (a, b) 를 고른다 — Pointer Events
  let dragging = false;
  const onPointerDown = (ev: PointerEvent) => {
    dragging = true;
    mapCanvas.setPointerCapture(ev.pointerId);
    stopMachine();
    statusEl.textContent = '';
    pick(ev);
  };
  const onPointerMove = (ev: PointerEvent) => {
    if (dragging) pick(ev);
  };
  const onPointerUp = (ev: PointerEvent) => {
    dragging = false;
    if (mapCanvas.hasPointerCapture(ev.pointerId)) mapCanvas.releasePointerCapture(ev.pointerId);
  };
  function pick(ev: PointerEvent): void {
    const [a, b] = map.eventToMath(ev);
    setParams({ a: clamp(a, A_RANGE), b: clamp(b, B_RANGE) });
  }
  mapCanvas.addEventListener('pointerdown', onPointerDown);
  mapCanvas.addEventListener('pointermove', onPointerMove);
  mapCanvas.addEventListener('pointerup', onPointerUp);
  mapCanvas.addEventListener('pointercancel', onPointerUp);

  // ── 기계에게 맡기기 (경사하강 애니메이션) ──
  function chosenLr(): keyof typeof ETA_CHOICES {
    const el = root.querySelector<HTMLInputElement>('input[name="lr"]:checked');
    const v = el?.value;
    return v === 'small' || v === 'tooBig' ? v : 'good';
  }

  function stopMachine(): void {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    running = false;
    machineBtn.textContent = t.machine;
  }

  function finish(reason: 'converged' | 'diverged' | 'steps'): void {
    stopMachine();
    if (reason === 'diverged') statusEl.textContent = t.diverged;
    else if (reason === 'converged') statusEl.textContent = t.converged;
    else statusEl.textContent = currentEta < 0.3 * etaMax ? t.slow : t.converged;
    render();
  }

  /** 한 스텝 진행. 끝났으면 이유를 돌려준다. */
  function stepOnce(): 'converged' | 'diverged' | 'steps' | null {
    const next = linearGdStep(train, p, currentEta);
    const L = linearLoss(train, next);
    if (!Number.isFinite(next.a) || !Number.isFinite(next.b) || !Number.isFinite(L) || Math.abs(next.a) > DIVERGE_BOUND || Math.abs(next.b) > DIVERGE_BOUND) {
      // 발산: 마지막 유한한 위치는 두고, 경로에 끊김 표시(NaN)를 남긴다
      path.push({ a: NaN, b: NaN });
      return 'diverged';
    }
    p = next;
    path.push(p);
    if (path.length > MAX_STEPS) return 'steps';
    if (Math.abs(p.a - star.a) < CONVERGE_TOL && Math.abs(p.b - star.b) < CONVERGE_TOL) return 'converged';
    return null;
  }

  function runMachine(choice: keyof typeof ETA_CHOICES = chosenLr(), animate = true): void {
    stopMachine();
    const radio = root.querySelector<HTMLInputElement>(`input[name="lr"][value="${choice}"]`);
    if (radio) radio.checked = true;
    currentEta = ETA_CHOICES[choice] * etaMax;
    path = [p];
    running = true;
    machineBtn.textContent = t.stop;
    statusEl.textContent = t.running;
    if (!animate) {
      let why: ReturnType<typeof stepOnce> = null;
      while (why === null) why = stepOnce();
      finish(why);
      return;
    }
    const frame = () => {
      if (!running) return;
      let why: ReturnType<typeof stepOnce> = null;
      for (let i = 0; i < STEPS_PER_FRAME && why === null; i++) why = stepOnce();
      render();
      if (why) {
        finish(why);
        return;
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  }

  machineBtn.addEventListener('click', () => {
    if (running) {
      stopMachine();
      statusEl.textContent = '';
      render();
    } else {
      runMachine();
    }
  });
  q<HTMLButtonElement>('next').addEventListener('click', () => {
    stopMachine();
    options.onNext({ a: p.a, b: p.b });
  });
  q<HTMLButtonElement>('skip').addEventListener('click', () => {
    stopMachine();
    (options.onSkip ?? (() => options.onNext({ a: star.a, b: star.b })))();
  });

  fit.onResize(drawFit);
  map.onResize(drawMap);
  render();

  return {
    destroy() {
      stopMachine();
      fit.destroy();
      map.destroy();
      root.innerHTML = '';
    },
    get params() {
      return { ...p };
    },
    runMachine,
  };
}

function clamp(v: number, r: readonly [number, number]): number {
  if (!Number.isFinite(v)) return r[0];
  return Math.max(r[0], Math.min(r[1], v));
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return '∞';
  return (v >= 0 ? '+' : '') + v.toFixed(2);
}
