/**
 * 데이터 생성 (SPEC 0절).
 *
 * 참 함수 g(x) = sin(2.5x) + 0.3x, 정의역 [-1, 1].
 * 학습 점 n개: 등간격 격자에 ±(간격의 30%) 균등 흔들기 → 양 끝 근처를 반드시 덮는다.
 * 검증(숨김) 점: x ~ U[min xᵢ, max xᵢ] — 학습 점 범위 밖에 두지 않는다 (외삽 오차 폭발 방지).
 * 잡음 ε ~ N(0, σ²), σ = 0.25.
 */
import { Rng } from './rng.ts';

export const TRUE_FN = (x: number): number => Math.sin(2.5 * x) + 0.3 * x;

export interface DataOptions {
  /** 학습 점 수 (기본 12) */
  n?: number;
  /** 잡음 표준편차 (기본 0.25) */
  sigma?: number;
  /** 검증 점 수 (기본 40) */
  nVal?: number;
  /** 격자 간격 대비 흔들기 비율 (기본 0.3) */
  jitter?: number;
}

export interface Points {
  readonly x: Float64Array;
  readonly y: Float64Array;
}

export interface Dataset {
  readonly seed: number;
  readonly sigma: number;
  readonly train: Points;
  /** 숨겨진 검증 점 */
  readonly val: Points;
}

export const DEFAULT_DATA: Required<DataOptions> = { n: 12, sigma: 0.25, nVal: 40, jitter: 0.3 };

export function generateDataset(seed: number, options: DataOptions = {}): Dataset {
  const { n, sigma, nVal, jitter } = { ...DEFAULT_DATA, ...options };
  if (n < 2) throw new Error('학습 점은 2개 이상이어야 합니다');
  if (!(sigma >= 0)) throw new Error('σ는 0 이상이어야 합니다');
  const rng = new Rng(seed);

  // 학습 x: 격자 + 흔들기, [-1, 1] 로 자른 뒤 정렬
  const h = 2 / (n - 1);
  const xs: number[] = [];
  for (let i = 0; i < n; i++) {
    const grid = -1 + i * h;
    const x = grid + rng.uniform(-jitter * h, jitter * h);
    xs.push(Math.max(-1, Math.min(1, x)));
  }
  xs.sort((p, q) => p - q);
  const trainX = Float64Array.from(xs);
  const trainY = new Float64Array(n);
  for (let i = 0; i < n; i++) trainY[i] = TRUE_FN(trainX[i]) + rng.normal(0, sigma);

  // 검증 x: 학습 점 범위 안에서 균등
  const lo = trainX[0];
  const hi = trainX[n - 1];
  const valX = new Float64Array(nVal);
  const valY = new Float64Array(nVal);
  for (let i = 0; i < nVal; i++) {
    valX[i] = rng.uniform(lo, hi);
    valY[i] = TRUE_FN(valX[i]) + rng.normal(0, sigma);
  }

  return { seed, sigma, train: { x: trainX, y: trainY }, val: { x: valX, y: valY } };
}

/** 평균제곱오차 (SPEC 기호 L) */
export function mse(pred: ArrayLike<number>, y: ArrayLike<number>): number {
  const n = y.length;
  if (n === 0) return NaN;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const r = pred[i] - y[i];
    s += r * r;
  }
  return s / n;
}
