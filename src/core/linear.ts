/**
 * 1단계: 직선 맞추기 (SPEC 1단계).
 *
 * 모델 ŷ = a x + b, 손실 L = (1/n) Σ (ŷᵢ − yᵢ)².
 * ∂L/∂a = (2/n) Σ rᵢ xᵢ,  ∂L/∂b = (2/n) Σ rᵢ,  rᵢ = ŷᵢ − yᵢ.
 * 헤세 행렬 H = (2/n) [[Σxᵢ², Σxᵢ], [Σxᵢ, n]] 은 상수 → 고정 학습률 경사하강법은
 * 0 < η < 2/λ_max(H) 일 때, 그리고 그때에만 수렴한다.
 */
import type { Points } from './data.ts';

export interface LinearParams {
  a: number;
  b: number;
}

export function linearPredict(p: LinearParams, x: number): number {
  return p.a * x + p.b;
}

export function linearLoss(data: Points, p: LinearParams): number {
  const { x, y } = data;
  const n = x.length;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const r = p.a * x[i] + p.b - y[i];
    s += r * r;
  }
  return s / n;
}

export function linearGrad(data: Points, p: LinearParams): LinearParams {
  const { x, y } = data;
  const n = x.length;
  let ga = 0;
  let gb = 0;
  for (let i = 0; i < n; i++) {
    const r = p.a * x[i] + p.b - y[i];
    ga += r * x[i];
    gb += r;
  }
  return { a: (2 / n) * ga, b: (2 / n) * gb };
}

/** 닫힌 해(최소제곱): a* = Sxy/Sxx, b* = ȳ − a* x̄ */
export function linearClosedForm(data: Points): LinearParams {
  const { x, y } = data;
  const n = x.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (x[i] - mx) * (x[i] - mx);
    sxy += (x[i] - mx) * (y[i] - my);
  }
  if (sxx === 0) throw new Error('x 가 모두 같아 직선을 정할 수 없습니다');
  const a = sxy / sxx;
  return { a, b: my - a * mx };
}

/** 헤세 행렬 (2/n)[[Σx², Σx],[Σx, n]] 의 성분 */
export function linearHessian(x: ArrayLike<number>): { h11: number; h12: number; h22: number } {
  const n = x.length;
  let sx = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sx += x[i];
    sxx += x[i] * x[i];
  }
  return { h11: (2 / n) * sxx, h12: (2 / n) * sx, h22: 2 };
}

/** 2×2 대칭행렬의 최대 고유값: (tr + √(tr² − 4 det)) / 2 */
export function linearLambdaMax(x: ArrayLike<number>): number {
  const { h11, h12, h22 } = linearHessian(x);
  const tr = h11 + h22;
  const det = h11 * h22 - h12 * h12;
  return (tr + Math.sqrt(Math.max(0, tr * tr - 4 * det))) / 2;
}

/** 수렴하는 고정 학습률의 상한 η_max = 2/λ_max */
export function linearEtaMax(x: ArrayLike<number>): number {
  return 2 / linearLambdaMax(x);
}

/** SPEC 의 학습률 선택지: 작음 0.1·η_max, 적당 0.6·η_max, 너무 큼 1.1·η_max */
export const ETA_CHOICES = { small: 0.1, good: 0.6, tooBig: 1.1 } as const;

/** 경사하강 한 스텝. NaN/Infinity 가 생겨도 던지지 않고 그대로 돌려준다(발산 시연용). */
export function linearGdStep(data: Points, p: LinearParams, eta: number): LinearParams {
  const g = linearGrad(data, p);
  return { a: p.a - eta * g.a, b: p.b - eta * g.b };
}

/** 여러 스텝의 경로. 결과 배열의 첫 원소는 시작점이다. */
export function linearGdPath(data: Points, start: LinearParams, eta: number, steps: number): LinearParams[] {
  const path: LinearParams[] = [start];
  let p = start;
  for (let i = 0; i < steps; i++) {
    p = linearGdStep(data, p, eta);
    path.push(p);
    if (!Number.isFinite(p.a) || !Number.isFinite(p.b)) break;
  }
  return path;
}
