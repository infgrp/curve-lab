/**
 * 신경망 그림 (3단계): 입력 x → 은닉 뉴런 H개(ReLU) → 출력 ŷ.
 * 선 굵기 = 가중치 크기, 색 = 부호(파랑 +, 빨강 −), 뉴런 채움 = 편향. 학습 중 매 프레임 다시 그린다.
 * H ≤ 8 이면 숫자(w, b, v)를 적고, 64 는 빽빽함만 보여 준다. 뉴런을 누르면 선택(hitTest).
 * 수학은 없다: Mlp 의 매개변수를 읽기만 한다.
 */
import type { Mlp } from '../core/mlp.ts';
import { COLORS } from './plot.ts';

const POS = '#0b6cff';
const NEG = '#d94a2b';
const GOLD = COLORS.path;

export class NetView {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private dpr = 1;
  private observer: ResizeObserver | null = null;
  private redraw: (() => void) | null = null;
  /** 마지막으로 그린 은닉 뉴런들의 중심 (hitTest 용) */
  private hidden: { x: number; y: number; r: number }[] = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D 캔버스를 만들 수 없습니다');
    this.ctx = ctx;
    this.resize();
  }

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
  }

  /** 포인터 위치(clientX/Y)에 있는 은닉 뉴런 번호, 없으면 null */
  hitTest(ev: { clientX: number; clientY: number }): number | null {
    const r = this.canvas.getBoundingClientRect();
    const px = ev.clientX - r.left;
    const py = ev.clientY - r.top;
    let best: number | null = null;
    let bestD = Infinity;
    this.hidden.forEach((n, i) => {
      const d = Math.hypot(px - n.x, py - n.y);
      if (d <= Math.max(24, n.r + 6) && d < bestD) {
        best = i;
        bestD = d;
      }
    });
    return best;
  }

  /**
   * @param net 학습 중/후의 신경망. null 이면 구조만 회색으로.
   * @param H 뉴런 수 (net 이 null 일 때 구조를 그리기 위해)
   * @param selected 강조할 은닉 뉴런 번호
   */
  draw(net: Mlp | null, H: number, selected: number | null, labels: { input: string; output: string; relu: string }): void {
    const { ctx, width: W, height: Hpx } = this;
    ctx.clearRect(0, 0, W, Hpx);
    const font = getComputedStyle(document.body).fontFamily;
    const padTop = 44;
    const padBottom = 28;
    const xIn = W * 0.13;
    const xHid = W * 0.5;
    const xOut = W * 0.87;
    const yMid = (padTop + Hpx - padBottom) / 2;
    const span = Hpx - padTop - padBottom;
    const big = H <= 8;
    const rHid = big ? Math.min(26, span / (2.6 * H)) : Math.max(3.5, Math.min(7, span / (2.2 * H)));
    const rIO = 28;
    const yHid = (j: number) => padTop + (span * (j + 0.5)) / H;
    this.hidden = Array.from({ length: H }, (_, j) => ({ x: xHid, y: yHid(j), r: rHid }));

    const wScale = big ? 2.2 : 0.9;
    const wMax = big ? 9 : 3.2;
    const edgeStyle = (w: number, sel: boolean) => {
      const a = Math.abs(w);
      ctx.lineWidth = Math.min(wMax, 0.6 + wScale * a);
      ctx.strokeStyle = sel ? GOLD : w >= 0 ? POS : NEG;
      ctx.globalAlpha = sel ? 1 : net ? Math.min(0.9, 0.25 + 0.35 * a) : 0.35;
    };

    // ── 선: 입력 → 은닉 (W₁), 은닉 → 출력 (W₂). 선택된 뉴런의 선은 마지막에 금색으로 ──
    const drawEdges = (onlySelected: boolean) => {
      for (let j = 0; j < H; j++) {
        const sel = selected === j;
        if (onlySelected !== sel) continue;
        const w1 = net ? net.W1[j] : 0.6;
        const w2 = net ? net.W2[j] : 0.6;
        const y = yHid(j);
        ctx.beginPath();
        edgeStyle(net ? w1 : 0, sel);
        if (!net) ctx.strokeStyle = '#8794a6';
        ctx.moveTo(xIn + rIO, yMid);
        ctx.lineTo(xHid - rHid, y);
        ctx.stroke();
        ctx.beginPath();
        edgeStyle(net ? w2 : 0, sel);
        if (!net) ctx.strokeStyle = '#8794a6';
        ctx.moveTo(xHid + rHid, y);
        ctx.lineTo(xOut - rIO, yMid);
        ctx.stroke();
      }
    };
    drawEdges(false);
    drawEdges(true);
    ctx.globalAlpha = 1;

    // ── 은닉 뉴런 ──
    for (let j = 0; j < H; j++) {
      const y = yHid(j);
      const b = net ? net.b1[j] : 0;
      const sel = selected === j;
      const tint = Math.min(1, Math.abs(b) / 1.5);
      ctx.beginPath();
      ctx.arc(xHid, y, rHid, 0, Math.PI * 2);
      ctx.fillStyle = !net ? 'rgba(135,148,166,.25)' : b >= 0 ? `rgba(11,108,255,${0.12 + 0.55 * tint})` : `rgba(217,74,43,${0.12 + 0.55 * tint})`;
      ctx.fill();
      ctx.lineWidth = sel ? 4 : big ? 2 : 1;
      ctx.strokeStyle = sel ? GOLD : 'rgba(19,33,58,.55)';
      ctx.stroke();
      if (big) {
        ctx.fillStyle = COLORS.text;
        ctx.font = `700 24px ${font}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(j + 1), xHid, y + 1);
      }
    }

    // ── 입력·출력 ──
    const io = (x: number, label: string) => {
      ctx.beginPath();
      ctx.arc(x, yMid, rIO, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = 'rgba(19,33,58,.6)';
      ctx.stroke();
      ctx.fillStyle = COLORS.text;
      ctx.font = `700 26px ${font}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x, yMid + 1);
    };
    io(xIn, labels.input);
    io(xOut, labels.output);

    // ── 숫자 (H ≤ 8): 선들이 벌어지는 뉴런 옆에 적는다. 왼쪽 = 입력 가중치 w, 오른쪽 = 출력 가중치 v, 위 = 편향 b ──
    if (big && net) {
      ctx.font = `24px ${font}`;
      ctx.textBaseline = 'middle';
      const tag = (text: string, x: number, yy: number, align: CanvasTextAlign, color: string) => {
        const wdt = ctx.measureText(text).width + 12;
        const left = align === 'left' ? x - 6 : align === 'right' ? x - wdt + 6 : x - wdt / 2;
        ctx.fillStyle = 'rgba(255,255,255,.88)';
        ctx.fillRect(left, yy - 15, wdt, 30);
        ctx.fillStyle = color;
        ctx.textAlign = align;
        ctx.fillText(text, x, yy + 1);
      };
      for (let j = 0; j < H; j++) {
        const y = yHid(j);
        const sel = selected === j;
        const cw = sel ? '#8a5a00' : net.W1[j] >= 0 ? '#0a56cc' : '#b23a1f';
        const cv = sel ? '#8a5a00' : net.W2[j] >= 0 ? '#0a56cc' : '#b23a1f';
        tag(fmt(net.W1[j]), xHid - rHid - 16, y, 'right', cw);
        tag(fmt(net.W2[j]), xHid + rHid + 16, y, 'left', cv);
        tag(`b ${fmt(net.b1[j])}`, xHid, y - rHid - 16, 'center', 'rgba(19,33,58,.75)');
      }
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(19,33,58,.75)';
      ctx.fillText(`${labels.output} 편향 ${fmt(net.b2)}`, xOut, yMid + rIO + 22);
    }

    // ── 열 이름 ──
    ctx.font = `24px ${font}`;
    ctx.fillStyle = 'rgba(19,33,58,.7)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(labels.input, xIn, 6);
    ctx.fillText(`${labels.relu} ${H}`, xHid, 6);
    ctx.fillText(labels.output, xOut, 6);
  }
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return '∞';
  return (v >= 0 ? '+' : '') + v.toFixed(2);
}
