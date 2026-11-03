/**
 * 3단계: 1 → H → 1 ReLU 신경망, 출력 선형, 전체 배치 (SPEC 3단계).
 *
 * 매개변수는 Float64Array 하나에 [W₁(H), b₁(H), W₂(H), b₂(1)] 순서로 둔다 (길이 3H + 1).
 * 초기화: W₁ ~ N(0, 2) (입력 1개 기준 He), b₁ = 0, W₂ ~ N(0, 2/H), b₂ = 0.
 * 역전파 버퍼(z, 활성값)는 학습 점 수가 바뀔 때만 다시 할당한다.
 */
import type { Points } from './data.ts';
import { Adam } from './optim.ts';
import type { Rng } from './rng.ts';

export const HIDDEN_CHOICES = [2, 8, 64] as const;

export class Mlp {
  readonly H: number;
  readonly params: Float64Array;
  readonly grad: Float64Array;
  // 매개변수 뷰 (params 의 부분 배열이므로 복사가 아니다)
  readonly W1: Float64Array;
  readonly b1: Float64Array;
  readonly W2: Float64Array;
  private z = new Float64Array(0); // n × H, 은닉 전활성
  private act = new Float64Array(0); // n × H, ReLU 출력
  private yhat = new Float64Array(0); // n

  constructor(H: number) {
    if (!Number.isInteger(H) || H < 1) throw new Error('은닉 뉴런 수는 1 이상의 정수여야 합니다');
    this.H = H;
    this.params = new Float64Array(3 * H + 1);
    this.grad = new Float64Array(3 * H + 1);
    this.W1 = this.params.subarray(0, H);
    this.b1 = this.params.subarray(H, 2 * H);
    this.W2 = this.params.subarray(2 * H, 3 * H);
  }

  get b2(): number {
    return this.params[3 * this.H];
  }

  /** SPEC 초기화 (시드 난수 사용) */
  static init(H: number, rng: Rng): Mlp {
    const net = new Mlp(H);
    for (let j = 0; j < H; j++) net.W1[j] = rng.normal(0, Math.SQRT2);
    for (let j = 0; j < H; j++) net.W2[j] = rng.normal(0, Math.sqrt(2 / H));
    return net;
  }

  predict(x: number): number {
    const { H, W1, b1, W2 } = this;
    let y = this.b2;
    for (let j = 0; j < H; j++) {
      const z = W1[j] * x + b1[j];
      if (z > 0) y += W2[j] * z;
    }
    return y;
  }

  predictInto(xs: ArrayLike<number>, out: Float64Array): Float64Array {
    for (let i = 0; i < xs.length; i++) out[i] = this.predict(xs[i]);
    return out;
  }

  loss(data: Points): number {
    const { x, y } = data;
    const n = x.length;
    let L = 0;
    for (let i = 0; i < n; i++) {
      const r = this.predict(x[i]) - y[i];
      L += r * r;
    }
    return L / n;
  }

  private ensureBuffers(n: number): void {
    if (this.yhat.length !== n) {
      this.z = new Float64Array(n * this.H);
      this.act = new Float64Array(n * this.H);
      this.yhat = new Float64Array(n);
    }
  }

  /** 손실을 돌려주고 this.grad 를 채운다 (역전파). 할당 없음. */
  lossAndGrad(data: Points): number {
    const { x, y } = data;
    const n = x.length;
    const { H, W1, b1, W2, grad } = this;
    this.ensureBuffers(n);
    const { z, act, yhat } = this;
    const b2 = this.b2;

    // 순전파
    let L = 0;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      let yi = b2;
      const base = i * H;
      for (let j = 0; j < H; j++) {
        const zj = W1[j] * xi + b1[j];
        z[base + j] = zj;
        const aj = zj > 0 ? zj : 0;
        act[base + j] = aj;
        yi += W2[j] * aj;
      }
      yhat[i] = yi;
      const r = yi - y[i];
      L += r * r;
    }
    L /= n;

    // 역전파: ∂L/∂ŷᵢ = 2 rᵢ / n
    grad.fill(0);
    const gW1 = grad.subarray(0, H);
    const gb1 = grad.subarray(H, 2 * H);
    const gW2 = grad.subarray(2 * H, 3 * H);
    let gb2 = 0;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      const g = (2 * (yhat[i] - y[i])) / n;
      gb2 += g;
      const base = i * H;
      for (let j = 0; j < H; j++) {
        gW2[j] += g * act[base + j];
        if (z[base + j] > 0) {
          const da = g * W2[j];
          gW1[j] += da * xi;
          gb1[j] += da;
        }
      }
    }
    grad[3 * H] = gb2;
    return L;
  }

  /** 다른 신경망의 매개변수를 복사해 온다 (같은 H) */
  copyFrom(other: Mlp): void {
    if (other.H !== this.H) throw new Error('H 가 다릅니다');
    this.params.set(other.params);
  }
}

/** 신경망 + Adam 을 묶은 학습기. UI 는 requestAnimationFrame 마다 `run(k)` 로 k epoch 씩 진행한다. */
export class MlpTrainer {
  readonly net: Mlp;
  readonly adam: Adam;
  readonly data: Points;
  epoch = 0;
  /** 마지막 epoch 의 학습 손실 */
  lastLoss = NaN;

  constructor(net: Mlp, data: Points, lr = 0.01) {
    this.net = net;
    this.adam = new Adam(net.params.length, { lr });
    this.data = data;
  }

  /** k epoch 진행하고 마지막 손실을 돌려준다 */
  run(k: number): number {
    for (let i = 0; i < k; i++) {
      this.lastLoss = this.net.lossAndGrad(this.data);
      this.adam.step(this.net.params, this.net.grad);
      this.epoch += 1;
    }
    return this.lastLoss;
  }
}
