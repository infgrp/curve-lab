/**
 * 2단계: 꺾은선 더하기 (SPEC 2단계).
 *
 * 모델 ŷ = a x + b + Σₖ sₖ·max(0, x − tₖ),  k = 1..K (기본 K = 3).
 *
 * 동치성: 은닉층 하나인 1입력 ReLU 신경망 Σ vₖ ReLU(wₖ x + βₖ) + c 의 각 항은
 *   wₖ > 0 → vₖwₖ·max(0, x − tₖ),  tₖ = −βₖ/wₖ
 *   wₖ < 0 → max(0, tₖ − x) = max(0, x − tₖ) − (x − tₖ) 이므로 꺾은선 하나 + 일차식
 *   wₖ = 0 → 상수
 * 으로 바뀌어, 이 모델은 "은닉 뉴런 K개인 ReLU 신경망 + 직선 항"과 같은 함수족이다 (fromReluNetwork).
 *
 * 기울기: ∂ŷ/∂sₖ = max(0, x − tₖ),  ∂ŷ/∂tₖ = −sₖ·1[x > tₖ].  x = tₖ 에서는 하위기울기 0.
 */
import type { Points } from './data.ts';
import type { LinearParams } from './linear.ts';

export interface HingeParams {
  a: number;
  b: number;
  /** 각 꺾은선의 기울기 변화량 sₖ */
  s: Float64Array;
  /** 각 꺾이는 점 tₖ */
  t: Float64Array;
}

export const DEFAULT_K = 3;

export function cloneHinge(p: HingeParams): HingeParams {
  return { a: p.a, b: p.b, s: Float64Array.from(p.s), t: Float64Array.from(p.t) };
}

/** 1단계에서 찾은 (a, b) 를 이어받아 초기화: tₖ 는 (−1, 1) 안에 등간격(K=3 이면 −0.5, 0, 0.5), sₖ = 0 */
export function hingeInitFromLinear(lin: LinearParams, K = DEFAULT_K): HingeParams {
  const t = new Float64Array(K);
  for (let k = 0; k < K; k++) t[k] = -1 + (2 * (k + 1)) / (K + 1);
  return { a: lin.a, b: lin.b, s: new Float64Array(K), t };
}

/** k번째 꺾은선 성분만: sₖ·max(0, x − tₖ) */
export function hingeComponent(p: HingeParams, k: number, x: number): number {
  return p.s[k] * Math.max(0, x - p.t[k]);
}

export function hingePredict(p: HingeParams, x: number): number {
  let y = p.a * x + p.b;
  const K = p.s.length;
  for (let k = 0; k < K; k++) {
    const d = x - p.t[k];
    if (d > 0) y += p.s[k] * d;
  }
  return y;
}

export function hingeLoss(data: Points, p: HingeParams): number {
  const { x, y } = data;
  const n = x.length;
  let L = 0;
  for (let i = 0; i < n; i++) {
    const r = hingePredict(p, x[i]) - y[i];
    L += r * r;
  }
  return L / n;
}

/** 기울기 (HingeParams 와 같은 모양). 결과는 새 객체. */
export function hingeGrad(data: Points, p: HingeParams): HingeParams {
  const { x, y } = data;
  const n = x.length;
  const K = p.s.length;
  const gs = new Float64Array(K);
  const gt = new Float64Array(K);
  let ga = 0;
  let gb = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const r = hingePredict(p, xi) - y[i];
    const c = (2 / n) * r;
    ga += c * xi;
    gb += c;
    for (let k = 0; k < K; k++) {
      const d = xi - p.t[k];
      if (d > 0) {
        gs[k] += c * d;
        gt[k] -= c * p.s[k];
      }
    }
  }
  return { a: ga, b: gb, s: gs, t: gt };
}

/**
 * 쉬고 있는 뉴런: 꺾이는 점이 모든 학습 점의 오른쪽(또는 그 위)에 있으면 어떤 점도 x > tₖ 가
 * 아니므로 sₖ, tₖ 의 기울기가 모두 0 이 되어 경사하강법으로는 움직이지 않는다.
 */
export function dormantNeurons(p: HingeParams, x: ArrayLike<number>): boolean[] {
  const K = p.s.length;
  const out: boolean[] = [];
  for (let k = 0; k < K; k++) {
    let active = false;
    for (let i = 0; i < x.length; i++) {
      if (x[i] > p.t[k]) {
        active = true;
        break;
      }
    }
    out.push(!active);
  }
  return out;
}

/** 경사하강 한 스텝 (새 객체 반환) */
export function hingeGdStep(data: Points, p: HingeParams, eta: number): HingeParams {
  const g = hingeGrad(data, p);
  const K = p.s.length;
  const s = new Float64Array(K);
  const t = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    s[k] = p.s[k] - eta * g.s[k];
    t[k] = p.t[k] - eta * g.t[k];
  }
  return { a: p.a - eta * g.a, b: p.b - eta * g.b, s, t };
}

// ── 평탄한 벡터 ↔ 매개변수 (최적화기·유한차분 검사용). 배치: [a, b, s₁..s_K, t₁..t_K] ──

export function hingeToVector(p: HingeParams): Float64Array {
  const K = p.s.length;
  const v = new Float64Array(2 + 2 * K);
  v[0] = p.a;
  v[1] = p.b;
  v.set(p.s, 2);
  v.set(p.t, 2 + K);
  return v;
}

export function hingeFromVector(v: Float64Array, K: number): HingeParams {
  if (v.length !== 2 + 2 * K) throw new Error(`벡터 길이 ${v.length} ≠ ${2 + 2 * K}`);
  return { a: v[0], b: v[1], s: v.slice(2, 2 + K), t: v.slice(2 + K, 2 + 2 * K) };
}

// ── ReLU 신경망 → 꺾은선 모델 변환 (동치성) ──

export interface ReluNetwork1D {
  /** 은닉 가중치 wₖ */
  w: ArrayLike<number>;
  /** 은닉 편향 βₖ */
  beta: ArrayLike<number>;
  /** 출력 가중치 vₖ */
  v: ArrayLike<number>;
  /** 출력 편향 c */
  c: number;
}

/**
 * Σ vₖ·max(0, wₖx + βₖ) + c 를 같은 함수의 꺾은선 모델로 바꾼다.
 * wₖ = 0 인 뉴런은 상수 vₖ·max(0, βₖ) 가 되어 b 에 흡수되고, 꺾은선 수가 하나 줄어든다.
 */
export function fromReluNetwork(net: ReluNetwork1D): HingeParams {
  const s: number[] = [];
  const t: number[] = [];
  let a = 0;
  let b = net.c;
  for (let k = 0; k < net.w.length; k++) {
    const w = net.w[k];
    const beta = net.beta[k];
    const v = net.v[k];
    if (w === 0) {
      b += v * Math.max(0, beta);
      continue;
    }
    const tk = -beta / w;
    if (w > 0) {
      // v·max(0, w(x − t)) = v·w·max(0, x − t)
      s.push(v * w);
    } else {
      // v·max(0, w(x − t)) = v·(−w)·max(0, t − x) = (−v w)·max(0, x − t) + v w (x − t)
      s.push(-v * w);
      a += v * w;
      b -= v * w * tk;
    }
    t.push(tk);
  }
  return { a, b, s: Float64Array.from(s), t: Float64Array.from(t) };
}

export function reluNetworkPredict(net: ReluNetwork1D, x: number): number {
  let y = net.c;
  for (let k = 0; k < net.w.length; k++) y += net.v[k] * Math.max(0, net.w[k] * x + net.beta[k]);
  return y;
}
