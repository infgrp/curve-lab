import { describe, expect, it } from 'vitest';
import { Adam, sgdStep } from '../src/core/optim.ts';

/** 장난감 문제 L(p) = (p₀ − 1)² + 2(p₁ + 1)²,  ∇L = (2(p₀ − 1), 4(p₁ + 1)) */
const toyGrad = (p: Float64Array) => Float64Array.of(2 * (p[0] - 1), 4 * (p[1] + 1));

describe('Adam', () => {
  it('(4) 한 스텝이 손으로 계산한 값과 일치한다', () => {
    // p = (0, 0) 에서 g = (−2, 4).  lr = 0.01, β₁ = 0.9, β₂ = 0.999, ε = 1e-8, t = 1.
    //   m = 0.1·g = (−0.2, 0.4),            v = 0.001·g² = (0.004, 0.016)
    //   m̂ = m/(1 − 0.9) = (−2, 4),           v̂ = v/(1 − 0.999) = (4, 16)
    //   갱신량 = lr·m̂/(√v̂ + ε) = 0.01·(−2/(2 + 1e-8), 4/(4 + 1e-8))
    //   p₀ = 0 + 0.01·2/(2 + 1e-8) = 0.01/(1 + 5e-9),   p₁ = 0 − 0.01·4/(4 + 1e-8) = −0.01/(1 + 2.5e-9)
    // (Adam 의 첫 스텝은 기울기 크기와 무관하게 거의 ±lr 만큼 움직인다는 점이 여기서 보인다.)
    const p = Float64Array.of(0, 0);
    const adam = new Adam(2);
    adam.step(p, toyGrad(p));
    expect(p[0]).toBeCloseTo(0.01 / (1 + 5e-9), 15);
    expect(p[1]).toBeCloseTo(-0.01 / (1 + 2.5e-9), 15);
    expect(adam.t).toBe(1);

    // 두 번째 스텝 (t = 2):
    //   g₂ = (2(p₀ − 1), 4(p₁ + 1)) ≈ (−1.98, 3.96)
    //   m = 0.9·(−0.2) + 0.1·(−1.98) = −0.378,  v = 0.999·0.004 + 0.001·1.98² = 0.0079160…
    //   m̂ = m/(1 − 0.81) = −1.98947…,  v̂ = v/(1 − 0.999²) = v/0.001999
    const g2 = toyGrad(p);
    const m0 = 0.9 * -0.2 + 0.1 * g2[0];
    const v0 = 0.999 * 0.004 + 0.001 * g2[0] * g2[0];
    const expected0 = p[0] - (0.01 * (m0 / (1 - 0.81))) / (Math.sqrt(v0 / (1 - 0.999 * 0.999)) + 1e-8);
    adam.step(p, g2);
    expect(p[0]).toBeCloseTo(expected0, 15);
    expect(adam.t).toBe(2);
  });

  it('장난감 문제에서 최솟점 (1, −1) 로 수렴한다', () => {
    const p = Float64Array.of(0, 0);
    const adam = new Adam(2, { lr: 0.05 });
    for (let i = 0; i < 2000; i++) adam.step(p, toyGrad(p));
    expect(p[0]).toBeCloseTo(1, 4);
    expect(p[1]).toBeCloseTo(-1, 4);
  });

  it('길이가 다르면 예외, reset 하면 처음부터', () => {
    const adam = new Adam(2);
    expect(() => adam.step(Float64Array.of(0, 0, 0), Float64Array.of(1, 1, 1))).toThrow();
    const p = Float64Array.of(0, 0);
    adam.step(p, toyGrad(p));
    adam.reset();
    const q = Float64Array.of(0, 0);
    adam.step(q, toyGrad(q));
    expect(Array.from(q)).toEqual(Array.from(p));
  });

  it('sgdStep 은 p − η g', () => {
    const p = Float64Array.of(1, 2);
    sgdStep(p, Float64Array.of(0.5, -1), 0.1);
    expect(Array.from(p)).toEqual([0.95, 2.1]);
  });
});
