/**
 * Adam 최적화기 (SPEC 3단계). 안내문에서는 "경사하강법을 개선한 방법"이라고 부른다.
 *
 * 매 스텝 t = 1, 2, …:
 *   m ← β₁ m + (1 − β₁) g,   v ← β₂ v + (1 − β₂) g²
 *   m̂ = m / (1 − β₁ᵗ),       v̂ = v / (1 − β₂ᵗ)
 *   p ← p − lr · m̂ / (√v̂ + ε)
 * reference/overfit_experiment.py 와 같은 순서(ε 은 √v̂ 뒤에 더함).
 * 메모리는 생성 시 한 번만 할당한다.
 */

export interface AdamOptions {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
}

export const DEFAULT_ADAM: Required<AdamOptions> = { lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8 };

export class Adam {
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  /** 지금까지 한 스텝 수 */
  t = 0;
  private readonly m: Float64Array;
  private readonly v: Float64Array;

  constructor(nParams: number, options: AdamOptions = {}) {
    const o = { ...DEFAULT_ADAM, ...options };
    this.lr = o.lr;
    this.beta1 = o.beta1;
    this.beta2 = o.beta2;
    this.eps = o.eps;
    this.m = new Float64Array(nParams);
    this.v = new Float64Array(nParams);
  }

  /** params 를 제자리에서 갱신한다. grad 는 같은 길이여야 한다. */
  step(params: Float64Array, grad: Float64Array): void {
    const n = params.length;
    if (grad.length !== n || this.m.length !== n) throw new Error('Adam: 길이가 맞지 않습니다');
    this.t += 1;
    const { lr, beta1, beta2, eps, m, v } = this;
    const c1 = 1 - Math.pow(beta1, this.t);
    const c2 = 1 - Math.pow(beta2, this.t);
    for (let i = 0; i < n; i++) {
      const g = grad[i];
      m[i] = beta1 * m[i] + (1 - beta1) * g;
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;
      params[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
    }
  }

  reset(): void {
    this.t = 0;
    this.m.fill(0);
    this.v.fill(0);
  }
}

/** 단순 경사하강 한 스텝 (제자리) */
export function sgdStep(params: Float64Array, grad: Float64Array, eta: number): void {
  for (let i = 0; i < params.length; i++) params[i] -= eta * grad[i];
}
