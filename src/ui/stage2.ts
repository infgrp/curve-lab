/**
 * 2단계 화면: 꺾은선 더하기 (SPEC 2단계 UI).
 *
 * 1단계에서 찾은 (a, b) 를 이어받아 ŷ = a x + b + Σ sₖ·max(0, x − tₖ), K = 3.
 * - 꺾이는 점마다 손잡이: 좌우로 끌면 tₖ, 위아래로 끌면 sₖ (Pointer Events, 손잡이 지름 48px 이상).
 *   손잡이가 겹치면 마지막에 잡은 것이 위로 온다.
 * - 각 꺾은선 성분은 옅은 색으로 따로, 합은 굵은 선으로.
 * - 뉴런 카드(슬라이더 t, s)와 "쉬고 있어요" 표시: 꺾이는 점이 모든 자료 점의 오른쪽이면 기울기가 0 이라 움직이지 않는다.
 * - "기계에게 맡기기": Adam 으로 (a, b, s, t) 를 함께 고친다(core/optim). 쉬는 뉴런은 그대로 남는 것을 숨기지 않는다.
 * 수학은 src/core/hinge.ts, src/core/optim.ts 가 한다.
 */
import type { Dataset } from '../core/data.ts';
import {
  DEFAULT_K,
  type HingeParams,
  cloneHinge,
  dormantNeurons,
  hingeComponent,
  hingeFromVector,
  hingeGrad,
  hingeInitFromLinear,
  hingeLoss,
  hingePredict,
  hingeToVector,
} from '../core/hinge.ts';
import { type LinearParams, linearLoss } from '../core/linear.ts';
import { Adam } from '../core/optim.ts';
import { legend, stageHeader } from './chrome.ts';
import { COLORS, Plot } from './plot.ts';
import { T } from './text.ko.ts';

export interface Stage2Options {
  onNext: (result: HingeParams) => void;
  onSkip?: () => void;
}

export interface Stage2Handle {
  destroy(): void;
  get params(): HingeParams;
  /** 테스트·점검용: 애니메이션 없이 끝까지 */
  runMachine(animate?: boolean): void;
  /** 테스트·점검용: 뉴런 k 의 (t, s) 를 직접 놓는다 */
  setNeuron(k: number, tk: number, sk: number): void;
}

export const NEURON_COLORS = ['#7c3aed', '#0e9f9a', '#c2417f'] as const;
const T_RANGE: readonly [number, number] = [-1.2, 1.2];
const S_RANGE: readonly [number, number] = [-8, 8];
const Y_RANGE: readonly [number, number] = [-2.2, 2.2];
const HANDLE_R = 19; // 그리는 반지름(px). 잡는 반지름은 아래
const HIT_R = 30; // 지름 60px ≥ 48px
/** 손잡이를 위아래로 끌 때: 그림 영역 높이 전체 = s 가 이만큼 변함 */
const S_PER_FULL_HEIGHT = 10;
const MAX_STEPS = 2500;
const STEPS_PER_FRAME = 6;
const ADAM_LR = 0.02;
const t = T.stage2;
const L = T.app.legend;

export function mountStage2(root: HTMLElement, data: Dataset, linear: LinearParams, options: Stage2Options): Stage2Handle {
  const train = data.train;
  const K = DEFAULT_K;
  const linLoss = linearLoss(train, linear);

  root.innerHTML = `
    <section class="stage stage2">
      ${stageHeader(2, t.title, t.lead, t.guide)}
      <div class="stage-body">
        <div class="glass panel">
          <div class="panel-head"><h2>${t.plotTitle}</h2>${legend([
            { kind: 'dot', color: COLORS.train, label: L.train },
            { kind: 'dash', color: '#8794a6', label: L.linear },
            { kind: 'line', color: NEURON_COLORS[0], label: L.component },
            { kind: 'line', color: COLORS.train, label: L.model },
          ])}</div>
          <canvas class="plot" data-role="fit"></canvas>
          <div class="controls">
            <div class="stats">
              <div class="card stat blue"><span class="k">${t.loss}</span><output class="v" data-role="loss"></output><span class="s" data-role="loss-sub"></span></div>
              <div class="card stat"><span class="k">${t.lossLinear}</span><output class="v" data-role="loss-lin">${linLoss.toFixed(3)}</output><span class="s"></span></div>
            </div>
          </div>
        </div>
        <div class="glass panel">
          <div class="panel-head"><h2>${t.neuronsTitle}</h2><span class="hint">${t.neuronsHint}</span></div>
          <div class="controls scroll" data-role="neurons"></div>
          <div class="controls">
            <p class="hint">${t.machineGuide}</p>
            <div class="row">
              <button class="btn" type="button" data-role="machine">▶ ${t.machine}</button>
              <button class="btn ghost" type="button" data-role="reset">${t.resetToLine}</button>
            </div>
            <p class="status" data-role="status" aria-live="polite"></p>
          </div>
        </div>
      </div>
      <footer class="stage-foot">
        <p class="bridge" data-role="bridge"></p>
        <div class="row">
          <button class="btn ghost" type="button" data-role="skip">${T.app.skip}</button>
          <button class="btn primary" type="button" data-role="next">${T.app.next} →</button>
        </div>
      </footer>
    </section>`;

  const q = <E extends Element>(role: string, from: ParentNode = root): E => {
    const el = from.querySelector<E>(`[data-role="${role}"]`);
    if (!el) throw new Error(`요소 없음: ${role}`);
    return el;
  };
  const fitCanvas = q<HTMLCanvasElement>('fit');
  const lossOut = q<HTMLOutputElement>('loss');
  const lossSub = q<HTMLSpanElement>('loss-sub');
  const statusEl = q<HTMLParagraphElement>('status');
  const bridgeEl = q<HTMLParagraphElement>('bridge');
  const machineBtn = q<HTMLButtonElement>('machine');
  const neuronsEl = q<HTMLDivElement>('neurons');

  // 뉴런 카드
  neuronsEl.innerHTML = Array.from({ length: K }, (_, k) => `
    <div class="card neuron" data-neuron="${k}">
      <span class="swatch" style="background:${NEURON_COLORS[k % NEURON_COLORS.length]}"></span>
      <div class="neuron-head"><span>${t.neuron(k + 1)}</span><span class="badge dot" data-role="badge"></span></div>
      <div class="sliders">
        <label class="slider"><span class="slider-name">${t.sliderT} <output data-role="t-out"></output></span>
          <input type="range" data-role="t" min="${T_RANGE[0]}" max="${T_RANGE[1]}" step="0.01" /></label>
        <label class="slider"><span class="slider-name">${t.sliderS} <output data-role="s-out"></output></span>
          <input type="range" data-role="s" min="${S_RANGE[0]}" max="${S_RANGE[1]}" step="0.05" /></label>
      </div>
    </div>`).join('');
  const cards = Array.from({ length: K }, (_, k) => {
    const card = neuronsEl.querySelector<HTMLDivElement>(`[data-neuron="${k}"]`)!;
    return {
      card,
      badge: q<HTMLSpanElement>('badge', card),
      tIn: q<HTMLInputElement>('t', card),
      sIn: q<HTMLInputElement>('s', card),
      tOut: q<HTMLOutputElement>('t-out', card),
      sOut: q<HTMLOutputElement>('s-out', card),
    };
  });

  // ── 상태 ──
  let p: HingeParams = hingeInitFromLinear(linear, K);
  let order: number[] = Array.from({ length: K }, (_, k) => k); // 뒤쪽이 위
  let raf = 0;
  let running = false;
  let adam: Adam | null = null;
  let vec: Float64Array | null = null;
  let lastGood: Float64Array | null = null; // 손실이 유한했던 마지막 매개변수
  let blewUp = false;
  let steps = 0;
  let lossHistory: number[] = [];

  const fit = new Plot(fitCanvas, { xRange: [-1.2, 1.2], yRange: Y_RANGE, xLabel: 'x', yLabel: 'y' });
  const model = (x: number) => hingePredict(p, x);

  function draw(): void {
    fit.clear();
    fit.drawAxes();
    fit.drawCurve((x) => linear.a * x + linear.b, { color: '#8794a6', width: 2, dash: [8, 6] });
    for (const k of order) {
      fit.drawCurve((x) => hingeComponent(p, k, x), { color: NEURON_COLORS[k % NEURON_COLORS.length], width: 2.5, alpha: 0.55 });
    }
    fit.drawResiduals(train.x, train.y, model, { color: 'rgba(217,74,43,.45)', width: 1.5 });
    fit.drawCurve(model, { color: COLORS.train, width: 4.5 });
    fit.drawPoints(train.x, train.y, { shape: 'circle', color: COLORS.train, radius: 9 });
    const rest = dormantNeurons(p, train.x);
    for (const k of order) {
      const y = hingePredict(p, p.t[k]);
      const color = NEURON_COLORS[k % NEURON_COLORS.length];
      fit.drawMarker(p.t[k], y, { radius: HANDLE_R, color, stroke: rest[k] ? COLORS.val : '#ffffff', strokeWidth: rest[k] ? 5 : 4 });
      fit.drawText(String(k + 1), p.t[k], y, { align: 'center', baseline: 'middle', color: '#fff', size: 24, dy: 1 });
      if (rest[k]) fit.drawText(t.resting, p.t[k], y, { align: 'right', baseline: 'middle', color: COLORS.val, size: 24, dx: -HANDLE_R - 8 });
    }
  }

  function syncCards(): void {
    const rest = dormantNeurons(p, train.x);
    for (let k = 0; k < K; k++) {
      const c = cards[k];
      c.tIn.value = String(clamp(p.t[k], T_RANGE));
      c.sIn.value = String(clamp(p.s[k], S_RANGE));
      c.tOut.value = fmt(p.t[k]);
      c.sOut.value = fmt(p.s[k]);
      c.badge.textContent = rest[k] ? t.resting : '';
      c.badge.className = `badge dot ${rest[k] ? 'rest' : ''}`.trim();
      c.badge.title = rest[k] ? t.restingHint : '';
      c.badge.style.visibility = rest[k] ? 'visible' : 'hidden';
    }
    const Lv = hingeLoss(train, p);
    lossOut.value = Number.isFinite(Lv) ? Lv.toFixed(3) : '∞';
    lossOut.classList.toggle('bad', !(Lv <= linLoss + 1e-9));
    const ratio = linLoss > 0 ? Lv / linLoss : 1;
    lossSub.textContent = Lv < linLoss ? `1단계 직선의 ${(ratio * 100).toFixed(0)}%` : '';
    bridgeEl.textContent = Lv < 0.9 * linLoss ? t.bridge : t.bridgeNotYet;
  }

  function render(): void {
    syncCards();
    draw();
  }

  function setStatus(text: string, tone: '' | 'good' | 'bad' = ''): void {
    statusEl.textContent = text;
    statusEl.className = `status ${tone}`.trim();
  }

  // ── 슬라이더 ──
  cards.forEach((c, k) => {
    c.tIn.addEventListener('input', () => {
      stopMachine();
      p.t[k] = Number(c.tIn.value);
      bringToTop(k);
      render();
    });
    c.sIn.addEventListener('input', () => {
      stopMachine();
      p.s[k] = Number(c.sIn.value);
      bringToTop(k);
      render();
    });
  });

  function bringToTop(k: number): void {
    order = order.filter((v) => v !== k);
    order.push(k);
  }

  // ── 손잡이 끌기 (Pointer Events) ──
  let drag: { k: number; pointerId: number; px: number; py: number; t0: number; s0: number } | null = null;

  function hitHandle(px: number, py: number): number | null {
    for (let i = order.length - 1; i >= 0; i--) {
      const k = order[i];
      const [hx, hy] = fit.toPx(p.t[k], hingePredict(p, p.t[k]));
      if (Math.hypot(px - hx, py - hy) <= HIT_R) return k;
    }
    return null;
  }

  const onPointerDown = (ev: PointerEvent) => {
    if (drag) return; // 끌고 있는 중에 다른 손가락이 닿아도 손잡이를 빼앗기지 않는다
    const r = fitCanvas.getBoundingClientRect();
    const px = ev.clientX - r.left;
    const py = ev.clientY - r.top;
    const k = hitHandle(px, py);
    if (k === null) return;
    stopMachine();
    setStatus('');
    bringToTop(k);
    drag = { k, pointerId: ev.pointerId, px, py, t0: p.t[k], s0: p.s[k] };
    fitCanvas.setPointerCapture(ev.pointerId);
    render();
  };
  const onPointerMove = (ev: PointerEvent) => {
    if (!drag || ev.pointerId !== drag.pointerId) return;
    const r = fitCanvas.getBoundingClientRect();
    const px = ev.clientX - r.left;
    const py = ev.clientY - r.top;
    const a = fit.area;
    const dxMath = ((px - drag.px) / (a.x1 - a.x0)) * (fit.xRange[1] - fit.xRange[0]);
    const dyFrac = (drag.py - py) / (a.y1 - a.y0); // 위로 끌면 양수
    p.t[drag.k] = clamp(drag.t0 + dxMath, T_RANGE);
    p.s[drag.k] = clamp(drag.s0 + dyFrac * S_PER_FULL_HEIGHT, S_RANGE);
    render();
  };
  const onPointerUp = (ev: PointerEvent) => {
    if (!drag || ev.pointerId !== drag.pointerId) return;
    if (fitCanvas.hasPointerCapture(ev.pointerId)) fitCanvas.releasePointerCapture(ev.pointerId);
    drag = null;
  };
  fitCanvas.addEventListener('pointerdown', onPointerDown);
  fitCanvas.addEventListener('pointermove', onPointerMove);
  fitCanvas.addEventListener('pointerup', onPointerUp);
  fitCanvas.addEventListener('pointercancel', onPointerUp);

  // ── 기계에게 맡기기 (Adam) ──
  function stopMachine(): void {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    running = false;
    adam = null;
    vec = null;
    machineBtn.textContent = `▶ ${t.machine}`;
  }

  /** 한 스텝. 끝났으면 true */
  function stepOnce(): boolean {
    if (!adam || !vec) return true;
    const g = hingeToVector(hingeGrad(train, p));
    lastGood = Float64Array.from(vec);
    adam.step(vec, g);
    p = hingeFromVector(vec, K);
    steps++;
    const Lv = hingeLoss(train, p);
    lossHistory.push(Lv);
    if (!Number.isFinite(Lv)) {
      // 값이 튀면 마지막으로 괜찮았던 자리로 되돌리고 멈춘다
      blewUp = true;
      if (lastGood) p = hingeFromVector(lastGood, K);
      return true;
    }
    if (steps >= MAX_STEPS) return true;
    if (lossHistory.length > 200) {
      const before = lossHistory[lossHistory.length - 201];
      if (before - Lv < 1e-6 * Math.max(before, 1e-9)) return true;
    }
    return false;
  }

  function finish(): void {
    const anyRest = dormantNeurons(p, train.x).some(Boolean);
    stopMachine();
    if (blewUp) setStatus(t.blewUp, 'bad');
    else setStatus(anyRest ? t.doneRest : t.done, 'good');
    render();
  }

  function runMachine(animate = true): void {
    stopMachine();
    vec = hingeToVector(p);
    adam = new Adam(vec.length, { lr: ADAM_LR });
    steps = 0;
    lossHistory = [];
    lastGood = null;
    blewUp = false;
    running = true;
    machineBtn.textContent = `■ ${t.stop}`;
    setStatus(t.running);
    if (!animate) {
      while (!stepOnce()) {
        /* 끝까지 */
      }
      finish();
      return;
    }
    const frame = () => {
      if (!running) return;
      let done = false;
      for (let i = 0; i < STEPS_PER_FRAME && !done; i++) done = stepOnce();
      render();
      if (done) {
        finish();
        return;
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  }

  machineBtn.addEventListener('click', () => {
    if (running) {
      stopMachine();
      setStatus('');
      render();
    } else runMachine();
  });
  q<HTMLButtonElement>('reset').addEventListener('click', () => {
    stopMachine();
    setStatus('');
    p = hingeInitFromLinear(linear, K);
    render();
  });
  q<HTMLButtonElement>('next').addEventListener('click', () => {
    stopMachine();
    options.onNext(cloneHinge(p));
  });
  q<HTMLButtonElement>('skip').addEventListener('click', () => {
    stopMachine();
    (options.onSkip ?? (() => options.onNext(cloneHinge(p))))();
  });

  fit.onResize(draw);
  render();

  return {
    destroy() {
      stopMachine();
      fit.destroy();
      root.innerHTML = '';
    },
    get params() {
      return cloneHinge(p);
    },
    runMachine,
    setNeuron(k, tk, sk) {
      stopMachine();
      p.t[k] = clamp(tk, T_RANGE);
      p.s[k] = clamp(sk, S_RANGE);
      bringToTop(k);
      render();
    },
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
