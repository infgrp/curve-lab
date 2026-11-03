import { describe, expect, it } from 'vitest';
import { DEFAULT_DATA, TRUE_FN, generateDataset, mse } from '../src/core/data.ts';

describe('generateDataset', () => {
  it('같은 시드면 같은 데이터', () => {
    const a = generateDataset(123);
    const b = generateDataset(123);
    expect(Array.from(a.train.x)).toEqual(Array.from(b.train.x));
    expect(Array.from(a.train.y)).toEqual(Array.from(b.train.y));
    expect(Array.from(a.val.x)).toEqual(Array.from(b.val.x));
    expect(Array.from(a.val.y)).toEqual(Array.from(b.val.y));
    const c = generateDataset(124);
    expect(Array.from(a.train.y)).not.toEqual(Array.from(c.train.y));
  });

  it('기본 크기: 학습 12개, 검증 40개, σ = 0.25', () => {
    const d = generateDataset(1);
    expect(d.train.x.length).toBe(12);
    expect(d.train.y.length).toBe(12);
    expect(d.val.x.length).toBe(40);
    expect(d.sigma).toBe(0.25);
  });

  it('학습 점이 [-1, 1] 안에 있고 양 끝 근처를 덮으며 오름차순', () => {
    const { n, jitter } = DEFAULT_DATA;
    const h = 2 / (n - 1);
    for (let seed = 0; seed < 50; seed++) {
      const { x } = generateDataset(seed).train;
      expect(x[0]).toBeGreaterThanOrEqual(-1);
      expect(x[n - 1]).toBeLessThanOrEqual(1);
      // 흔들기 폭이 0.3h 이므로 첫 점은 -1 + 0.3h 보다 왼쪽, 마지막 점은 1 - 0.3h 보다 오른쪽
      expect(x[0]).toBeLessThanOrEqual(-1 + jitter * h + 1e-12);
      expect(x[n - 1]).toBeGreaterThanOrEqual(1 - jitter * h - 1e-12);
      for (let i = 1; i < n; i++) expect(x[i]).toBeGreaterThanOrEqual(x[i - 1]);
    }
  });

  it('검증 점의 x 는 학습 점 x 범위 안에 있다 (외삽 없음)', () => {
    for (let seed = 0; seed < 50; seed++) {
      const d = generateDataset(seed);
      const lo = d.train.x[0];
      const hi = d.train.x[d.train.x.length - 1];
      for (const x of d.val.x) {
        expect(x).toBeGreaterThanOrEqual(lo);
        expect(x).toBeLessThanOrEqual(hi);
      }
    }
  });

  it('y 는 참 함수 + 잡음: 잔차의 표본분산이 σ² 근처 (많은 시드를 합쳐서)', () => {
    let s2 = 0;
    let cnt = 0;
    for (let seed = 0; seed < 200; seed++) {
      const d = generateDataset(seed);
      for (let i = 0; i < d.train.x.length; i++) {
        const r = d.train.y[i] - TRUE_FN(d.train.x[i]);
        s2 += r * r;
        cnt++;
      }
    }
    // 잔차 2400개의 제곱평균 ≈ σ² = 0.0625, 표준오차 ≈ σ²·√(2/2400) ≈ 0.0018 → 4배
    expect(Math.abs(s2 / cnt - 0.0625)).toBeLessThan(4 * 0.0625 * Math.sqrt(2 / cnt));
  });

  it('옵션: 점 수와 σ 를 바꿀 수 있고, σ = 0 이면 잡음이 없다', () => {
    const d = generateDataset(9, { n: 8, sigma: 0, nVal: 5 });
    expect(d.train.x.length).toBe(8);
    expect(d.val.x.length).toBe(5);
    for (let i = 0; i < 8; i++) expect(d.train.y[i]).toBeCloseTo(TRUE_FN(d.train.x[i]), 14);
  });
});

describe('mse', () => {
  it('손으로 계산한 값과 같다', () => {
    expect(mse([1, 2, 3], [1, 1, 1])).toBeCloseTo((0 + 1 + 4) / 3, 14);
    expect(mse([], [])).toBeNaN();
  });
});
