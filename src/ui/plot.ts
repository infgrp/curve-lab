/**
 * 캔버스 그리기 공통부 (PROCEDURE Phase 3).
 *
 * - 수학 좌표(x ∈ xRange, y ∈ yRange) ↔ 캔버스 CSS 픽셀 좌표 변환
 * - 축·눈금, 점(원/다이아몬드 — 색만으로 구분하지 않는다), 곡선(픽셀 단위 샘플링), 잔차 선분,
 *   등고선(core/contour 의 marching squares), 꺾은선 경로, 표식
 * - devicePixelRatio 반영, 크기가 바뀌면 다시 그림 (ResizeObserver)
 * 수학은 여기서 다시 구현하지 않는다: 등고선 선분은 core 가 계산한다.
 */
import { type Grid, marchingSquares } from '../core/contour.ts';

export type Range = readonly [number, number];

export interface Padding {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export interface PlotOptions {
  xRange: Range;
  yRange: Range;
  /** 축 라벨 글자 크기(px). 키오스크 최소 24 */
  fontSize?: number;
  padding?: Partial<Padding>;
  /** 축 라벨 */
  xLabel?: string;
  yLabel?: string;
}

export type PointShape = 'circle' | 'diamond';

export interface PointStyle {
  shape?: PointShape;
  color?: string;
  /** 반지름(px) */
  radius?: number;
  stroke?: string;
  strokeWidth?: number;
}

export interface LineStyle {
  color?: string;
  width?: number;
  dash?: number[];
  alpha?: number;
}

export const COLORS = {
  axis: '#55657a',
  grid: 'rgba(21,34,58,.10)',
  text: '#15223a',
  train: '#0b6cff',
  val: '#d94a2b',
  curve: '#15223a',
  residual: 'rgba(217,74,43,.8)',
  contour: 'rgba(21,34,58,.45)',
  path: '#ffb000',
} as const;

const DEFAULT_PADDING: Padding = { left: 72, right: 20, top: 20, bottom: 56 };

export class Plot {
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  xRange: Range;
  yRange: Range;
  fontSize: number;
  padding: Padding;
  xLabel: string;
  yLabel: string;
  /** CSS 픽셀 기준 캔버스 크기 */
  width = 0;
  height = 0;
  private dpr = 1;
  private observer: ResizeObserver | null = null;
  private redraw: (() => void) | null = null;

  constructor(canvas: HTMLCanvasElement, options: PlotOptions) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D 캔버스를 만들 수 없습니다');
    this.ctx = ctx;
    this.xRange = options.xRange;
    this.yRange = options.yRange;
    this.fontSize = options.fontSize ?? 24;
    this.padding = { ...DEFAULT_PADDING, ...options.padding };
    this.xLabel = options.xLabel ?? '';
    this.yLabel = options.yLabel ?? '';
    this.resize();
  }

  // ── 크기 ──

  /** CSS 크기를 읽어 백킹 스토어를 devicePixelRatio 배로 맞춘다. 크기가 바뀌었으면 true */
  resize(): boolean {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const changed = w !== this.width || h !== this.height || dpr !== this.dpr;
    if (changed) {
      this.width = w;
      this.height = h;
      this.dpr = dpr;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return changed;
  }

  /** 크기가 바뀔 때마다 redraw 를 부른다 (바로 한 번 부름) */
  onResize(redraw: () => void): void {
    this.redraw = redraw;
    this.observer?.disconnect();
    this.observer = new ResizeObserver(() => {
      if (this.resize()) this.redraw?.();
    });
    this.observer.observe(this.canvas);
    redraw();
  }

  destroy(): void {
    this.observer?.disconnect();
    this.observer = null;
    this.redraw = null;
  }

  // ── 좌표 ──

  /** 그림 영역 (CSS px) */
  get area(): { x0: number; y0: number; x1: number; y1: number } {
    return {
      x0: this.padding.left,
      y0: this.padding.top,
      x1: this.width - this.padding.right,
      y1: this.height - this.padding.bottom,
    };
  }

  toPx(x: number, y: number): [number, number] {
    const a = this.area;
    const px = a.x0 + ((x - this.xRange[0]) / (this.xRange[1] - this.xRange[0])) * (a.x1 - a.x0);
    const py = a.y1 - ((y - this.yRange[0]) / (this.yRange[1] - this.yRange[0])) * (a.y1 - a.y0);
    return [px, py];
  }

  fromPx(px: number, py: number): [number, number] {
    const a = this.area;
    const x = this.xRange[0] + ((px - a.x0) / (a.x1 - a.x0)) * (this.xRange[1] - this.xRange[0]);
    const y = this.yRange[0] + ((a.y1 - py) / (a.y1 - a.y0)) * (this.yRange[1] - this.yRange[0]);
    return [x, y];
  }

  /** 포인터 이벤트의 clientX/Y → 수학 좌표 */
  eventToMath(ev: { clientX: number; clientY: number }): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return this.fromPx(ev.clientX - r.left, ev.clientY - r.top);
  }

  // ── 그리기 ──

  clear(): void {
    this.ctx.clearRect(0, 0, this.width, this.height);
  }

  /** 그림 영역 밖을 그리지 않도록 잘라낸 채 fn 실행 */
  clipped(fn: () => void): void {
    const { ctx } = this;
    const a = this.area;
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x0, a.y0, a.x1 - a.x0, a.y1 - a.y0);
    ctx.clip();
    try {
      fn();
    } finally {
      ctx.restore();
    }
  }

  drawAxes(opts: { xTicks?: number[]; yTicks?: number[]; grid?: boolean; format?: (v: number) => string } = {}): void {
    const { ctx } = this;
    const a = this.area;
    const fmt = opts.format ?? ((v: number) => (Math.abs(v) < 1e-12 ? '0' : String(+v.toFixed(3))));
    const xTicks = opts.xTicks ?? niceTicks(this.xRange, 6);
    const yTicks = opts.yTicks ?? niceTicks(this.yRange, 5);
    ctx.save();
    ctx.font = `${this.fontSize}px system-ui, sans-serif`;
    ctx.fillStyle = COLORS.text;
    ctx.lineWidth = 1;
    // 격자
    if (opts.grid !== false) {
      ctx.strokeStyle = COLORS.grid;
      for (const x of xTicks) {
        const [px] = this.toPx(x, 0);
        if (px < a.x0 - 0.5 || px > a.x1 + 0.5) continue;
        ctx.beginPath();
        ctx.moveTo(Math.round(px) + 0.5, a.y0);
        ctx.lineTo(Math.round(px) + 0.5, a.y1);
        ctx.stroke();
      }
      for (const y of yTicks) {
        const [, py] = this.toPx(0, y);
        if (py < a.y0 - 0.5 || py > a.y1 + 0.5) continue;
        ctx.beginPath();
        ctx.moveTo(a.x0, Math.round(py) + 0.5);
        ctx.lineTo(a.x1, Math.round(py) + 0.5);
        ctx.stroke();
      }
    }
    // 축선: x=0, y=0 이 범위 안에 있으면 그 선, 아니면 테두리
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1.5;
    const [ox, oy] = this.toPx(0, 0);
    const axisX = oy >= a.y0 && oy <= a.y1 ? oy : a.y1;
    const axisY = ox >= a.x0 && ox <= a.x1 ? ox : a.x0;
    ctx.beginPath();
    ctx.moveTo(a.x0, axisX);
    ctx.lineTo(a.x1, axisX);
    ctx.moveTo(axisY, a.y0);
    ctx.lineTo(axisY, a.y1);
    ctx.stroke();
    // 눈금 라벨 (항상 아래·왼쪽 가장자리에)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const x of xTicks) {
      const [px] = this.toPx(x, 0);
      if (px < a.x0 - 0.5 || px > a.x1 + 0.5) continue;
      ctx.fillText(fmt(x), px, a.y1 + 8);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const y of yTicks) {
      const [, py] = this.toPx(0, y);
      if (py < a.y0 - 0.5 || py > a.y1 + 0.5) continue;
      ctx.fillText(fmt(y), a.x0 - 10, py);
    }
    // 축 이름
    if (this.xLabel) {
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(this.xLabel, a.x1, a.y1 - 6);
    }
    if (this.yLabel) {
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(this.yLabel, a.x0 + 8, a.y0 + 4);
    }
    ctx.restore();
  }

  drawPoints(xs: ArrayLike<number>, ys: ArrayLike<number>, style: PointStyle = {}): void {
    const { ctx } = this;
    const shape = style.shape ?? 'circle';
    const r = style.radius ?? 7;
    ctx.save();
    ctx.fillStyle = style.color ?? COLORS.train;
    ctx.strokeStyle = style.stroke ?? '#ffffff';
    ctx.lineWidth = style.strokeWidth ?? 2;
    for (let i = 0; i < xs.length; i++) {
      const [px, py] = this.toPx(xs[i], ys[i]);
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      ctx.beginPath();
      if (shape === 'circle') {
        ctx.arc(px, py, r, 0, Math.PI * 2);
      } else {
        const d = r * 1.25;
        ctx.moveTo(px, py - d);
        ctx.lineTo(px + d, py);
        ctx.lineTo(px, py + d);
        ctx.lineTo(px - d, py);
        ctx.closePath();
      }
      ctx.fill();
      if (ctx.lineWidth > 0) ctx.stroke();
    }
    ctx.restore();
  }

  /** 함수를 그림 영역의 픽셀 열마다 샘플링해 곡선을 그린다. 유한하지 않은 값에서는 선을 끊는다. */
  drawCurve(f: (x: number) => number, style: LineStyle = {}, xFrom?: number, xTo?: number): void {
    const { ctx } = this;
    const a = this.area;
    const [pxFrom] = xFrom === undefined ? [a.x0] : this.toPx(xFrom, 0);
    const [pxTo] = xTo === undefined ? [a.x1] : this.toPx(xTo, 0);
    const start = Math.max(a.x0, Math.min(pxFrom, pxTo));
    const end = Math.min(a.x1, Math.max(pxFrom, pxTo));
    this.clipped(() => {
      ctx.strokeStyle = style.color ?? COLORS.curve;
      ctx.lineWidth = style.width ?? 3;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.globalAlpha = style.alpha ?? 1;
      ctx.setLineDash(style.dash ?? []);
      ctx.beginPath();
      let pen = false;
      for (let px = start; px <= end; px += 1) {
        const [x] = this.fromPx(px, 0);
        const y = f(x);
        if (!Number.isFinite(y)) {
          pen = false;
          continue;
        }
        const [, py] = this.toPx(x, y);
        // 화면 밖 아주 큰 값은 잘라내 캔버스가 멈추지 않게 한다
        const pyc = Math.max(-1e5, Math.min(1e5, py));
        if (pen) ctx.lineTo(px, pyc);
        else ctx.moveTo(px, pyc);
        pen = true;
      }
      ctx.stroke();
    });
  }

  /** 잔차 선분: 각 점 (xᵢ, yᵢ) 에서 곡선 값 f(xᵢ) 까지의 세로 선 */
  drawResiduals(xs: ArrayLike<number>, ys: ArrayLike<number>, f: (x: number) => number, style: LineStyle = {}): void {
    const { ctx } = this;
    this.clipped(() => {
      ctx.strokeStyle = style.color ?? COLORS.residual;
      ctx.lineWidth = style.width ?? 2;
      ctx.setLineDash(style.dash ?? []);
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const yh = f(xs[i]);
        if (!Number.isFinite(yh)) continue;
        const [px, py] = this.toPx(xs[i], ys[i]);
        const [, ph] = this.toPx(xs[i], yh);
        ctx.moveTo(px, py);
        ctx.lineTo(px, Math.max(-1e5, Math.min(1e5, ph)));
      }
      ctx.stroke();
    });
  }

  /** 등고선: 격자 값과 수준들을 받아 marching squares 로 그린다 */
  drawContours(grid: Grid, levels: ArrayLike<number>, style: LineStyle = {}): void {
    const { ctx } = this;
    this.clipped(() => {
      ctx.strokeStyle = style.color ?? COLORS.contour;
      ctx.lineWidth = style.width ?? 1.5;
      ctx.setLineDash(style.dash ?? []);
      ctx.globalAlpha = style.alpha ?? 1;
      for (let k = 0; k < levels.length; k++) {
        ctx.beginPath();
        for (const [x1, y1, x2, y2] of marchingSquares(grid, levels[k])) {
          const [p1x, p1y] = this.toPx(x1, y1);
          const [p2x, p2y] = this.toPx(x2, y2);
          ctx.moveTo(p1x, p1y);
          ctx.lineTo(p2x, p2y);
        }
        ctx.stroke();
      }
    });
  }

  /** 꺾은선 경로 (예: 경사하강 궤적). 유한하지 않은 점에서 끊는다. */
  drawPath(points: ArrayLike<{ x: number; y: number }>, style: LineStyle = {}, dots = true): void {
    const { ctx } = this;
    this.clipped(() => {
      ctx.strokeStyle = style.color ?? COLORS.path;
      ctx.fillStyle = style.color ?? COLORS.path;
      ctx.lineWidth = style.width ?? 2.5;
      ctx.setLineDash(style.dash ?? []);
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) {
          pen = false;
          continue;
        }
        const [px, py] = this.toPx(p.x, p.y);
        const cx = Math.max(-1e5, Math.min(1e5, px));
        const cy = Math.max(-1e5, Math.min(1e5, py));
        if (pen) ctx.lineTo(cx, cy);
        else ctx.moveTo(cx, cy);
        pen = true;
      }
      ctx.stroke();
      if (dots) {
        for (let i = 0; i < points.length; i++) {
          const p = points[i];
          if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
          const [px, py] = this.toPx(p.x, p.y);
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });
  }

  /** 현재 위치 같은 강조 표식 (고리 있는 점) */
  drawMarker(x: number, y: number, style: PointStyle = {}): void {
    const { ctx } = this;
    const [px, py] = this.toPx(x, y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) return;
    const r = style.radius ?? 9;
    ctx.save();
    ctx.fillStyle = style.color ?? COLORS.path;
    ctx.strokeStyle = style.stroke ?? '#ffffff';
    ctx.lineWidth = style.strokeWidth ?? 3;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  /** 그림 영역 안의 글자 */
  drawText(text: string, x: number, y: number, opts: { align?: CanvasTextAlign; baseline?: CanvasTextBaseline; color?: string; size?: number; dx?: number; dy?: number } = {}): void {
    const { ctx } = this;
    const [px, py] = this.toPx(x, y);
    ctx.save();
    ctx.font = `${opts.size ?? this.fontSize}px system-ui, sans-serif`;
    ctx.fillStyle = opts.color ?? COLORS.text;
    ctx.textAlign = opts.align ?? 'left';
    ctx.textBaseline = opts.baseline ?? 'bottom';
    ctx.fillText(text, px + (opts.dx ?? 0), py + (opts.dy ?? 0));
    ctx.restore();
  }
}

/** 보기 좋은 눈금 (1, 2, 5 × 10ⁿ 간격) */
export function niceTicks(range: Range, target = 6): number[] {
  const span = range[1] - range[0];
  if (!(span > 0)) return [];
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  let step = 10 * mag;
  for (const m of [1, 2, 5]) {
    if (m * mag >= raw) {
      step = m * mag;
      break;
    }
  }
  const out: number[] = [];
  const first = Math.ceil(range[0] / step - 1e-9);
  const last = Math.floor(range[1] / step + 1e-9);
  for (let k = first; k <= last; k++) out.push(+(k * step).toFixed(10));
  return out;
}
