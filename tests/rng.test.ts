import { describe, expect, it } from 'vitest';
import { Rng, mulberry32, seedFrom } from '../src/core/rng.ts';

describe('mulberry32', () => {
  it('같은 시드면 같은 수열, 다른 시드면 다른 수열', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    const c = mulberry32(43);
    const sa = Array.from({ length: 20 }, () => a());
    const sb = Array.from({ length: 20 }, () => b());
    const sc = Array.from({ length: 20 }, () => c());
    expect(sa).toEqual(sb);
    expect(sa).not.toEqual(sc);
  });

  it('값은 [0, 1) 안에 있고 고르게 퍼진다', () => {
    const r = mulberry32(7);
    const bins = new Array<number>(10).fill(0);
    const N = 100_000;
    for (let i = 0; i < N; i++) {
      const u = r();
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(1);
      bins[Math.floor(u * 10)]++;
    }
    // 각 칸의 기대 도수 10000, 표준편차 √(N·0.1·0.9) ≈ 95 → 4σ ≈ 380 안
    for (const c of bins) expect(Math.abs(c - N / 10)).toBeLessThan(380);
  });
});

describe('Rng.normal (Box–Muller)', () => {
  it('10만 개의 표본평균 ≈ 0, 표본분산 ≈ 1', () => {
    const rng = new Rng(2024);
    const N = 100_000;
    let s = 0;
    let s2 = 0;
    for (let i = 0; i < N; i++) {
      const z = rng.normal();
      s += z;
      s2 += z * z;
    }
    const mean = s / N;
    const variance = s2 / N - mean * mean;
    // 표본평균의 표준오차 = 1/√N ≈ 0.00316 → 4배 = 0.0126
    expect(Math.abs(mean)).toBeLessThan(4 / Math.sqrt(N));
    // 표본분산의 표준오차 ≈ √(2/N) ≈ 0.00447 (정규분포에서 Var(s²) ≈ 2σ⁴/N) → 4배 = 0.0179
    expect(Math.abs(variance - 1)).toBeLessThan(4 * Math.sqrt(2 / N));
  });

  it('평균·표준편차 인자가 반영된다', () => {
    const rng = new Rng(5);
    const N = 50_000;
    let s = 0;
    for (let i = 0; i < N; i++) s += rng.normal(3, 0.5);
    expect(Math.abs(s / N - 3)).toBeLessThan(4 * 0.5 / Math.sqrt(N));
  });

  it('같은 시드면 normal 도 같은 수열 (보관된 짝 포함)', () => {
    const a = new Rng(99);
    const b = new Rng(99);
    for (let i = 0; i < 11; i++) expect(a.normal()).toBe(b.normal());
  });
});

describe('seedFrom', () => {
  it('32비트 부호 없는 정수를 돌려주고 입력이 다르면 보통 다르다', () => {
    const s1 = seedFrom(1_730_000_000_000);
    const s2 = seedFrom(1_730_000_000_001);
    const s3 = seedFrom(1_730_000_000_000, 12345);
    for (const s of [s1, s2, s3]) {
      expect(Number.isInteger(s)).toBe(true);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThan(2 ** 32);
    }
    expect(s1).not.toBe(s2);
    expect(s1).not.toBe(s3);
  });
});
