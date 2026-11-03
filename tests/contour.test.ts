import { describe, expect, it } from 'vitest';
import { gridMinMax, logLevels, makeGrid, marchingSquares } from '../src/core/contour.ts';

describe('makeGrid / logLevels', () => {
  it('격자 값은 f(xs[i], ys[j]) 를 행 우선으로 담는다', () => {
    const g = makeGrid([0, 2], [10, 11], 3, 2, (x, y) => x + 100 * y);
    expect(Array.from(g.xs)).toEqual([0, 1, 2]);
    expect(Array.from(g.ys)).toEqual([10, 11]);
    expect(Array.from(g.v)).toEqual([1000, 1001, 1002, 1100, 1101, 1102]);
    expect(gridMinMax(g)).toEqual({ min: 1000, max: 1102 });
    expect(() => makeGrid([0, 1], [0, 1], 1, 2, () => 0)).toThrow();
  });

  it('로그 간격 수준은 등비수열이고 양 끝을 제외한다', () => {
    const lv = logLevels(1, 1000, 2);
    expect(lv).toHaveLength(2);
    expect(lv[0]).toBeCloseTo(10, 10);
    expect(lv[1]).toBeCloseTo(100, 10);
    expect(logLevels(0, 1, 3)).toEqual([]);
    expect(logLevels(2, 1, 3)).toEqual([]);
  });
});

describe('marchingSquares', () => {
  const circle = makeGrid([-2, 2], [-2, 2], 81, 81, (x, y) => x * x + y * y);

  it('f = x² + y² 의 수준 1 등고선은 반지름 1 원 위에 있고 둘레가 2π 에 가깝다', () => {
    const segs = marchingSquares(circle, 1);
    expect(segs.length).toBeGreaterThan(50);
    let length = 0;
    for (const [x1, y1, x2, y2] of segs) {
      // 끝점은 셀 변 위에서 선형 보간한 값이므로 셀 크기(0.05)의 몇 분의 일 안에서 원 위에 있다
      expect(Math.abs(Math.hypot(x1, y1) - 1)).toBeLessThan(0.01);
      expect(Math.abs(Math.hypot(x2, y2) - 1)).toBeLessThan(0.01);
      length += Math.hypot(x2 - x1, y2 - y1);
    }
    expect(Math.abs(length - 2 * Math.PI)).toBeLessThan(0.02);
  });

  it('등고선 선분들은 끝과 끝이 이어진다 (각 끝점이 다른 선분의 끝점과 만남)', () => {
    const segs = marchingSquares(circle, 1);
    const key = (x: number, y: number) => `${x.toFixed(9)},${y.toFixed(9)}`;
    const count = new Map<string, number>();
    for (const [x1, y1, x2, y2] of segs) {
      count.set(key(x1, y1), (count.get(key(x1, y1)) ?? 0) + 1);
      count.set(key(x2, y2), (count.get(key(x2, y2)) ?? 0) + 1);
    }
    // 닫힌 곡선이므로 모든 끝점이 정확히 두 번 나타난다
    for (const c of count.values()) expect(c).toBe(2);
  });

  it('최솟값보다 낮거나 최댓값보다 높은 수준에는 선분이 없다', () => {
    expect(marchingSquares(circle, -1)).toEqual([]);
    expect(marchingSquares(circle, 100)).toEqual([]);
  });

  it('안장 셀(마주 보는 꼭짓점만 높음)은 선분 두 개를 내고 중심값으로 방향을 정한다', () => {
    // 2×2 격자 하나. v 는 행 우선이라 v[0]=(0,0), v[1]=(1,0), v[2]=(0,1), v[3]=(1,1).
    // 마주 보는 (0,0)과 (1,1)을 1(높음), 나머지를 0(낮음)으로, 수준 0.5. 중심 0.5 는 수준보다 크지 않으므로
    // 낮은 쪽끼리 이어지는 방향(선분 두 개가 각각 왼쪽-아래, 오른쪽-위 모서리를 자름)
    const g = { xs: Float64Array.of(0, 1), ys: Float64Array.of(0, 1), v: Float64Array.of(1, 0, 0, 1) };
    const segs = marchingSquares(g, 0.5);
    expect(segs).toHaveLength(2);
    for (const [x1, y1, x2, y2] of segs) {
      expect(Math.abs(Math.abs(x2 - x1) - 0.5)).toBeLessThan(1e-12);
      expect(Math.abs(Math.abs(y2 - y1) - 0.5)).toBeLessThan(1e-12);
    }
    // 높은 꼭짓점을 4 로 올리면 중심 (4+0+0+4)/4 = 2 > 0.5 → 높은 쪽끼리 이어져 반대 방향
    const g2 = { xs: g.xs, ys: g.ys, v: Float64Array.of(4, 0, 0, 4) };
    const s2 = marchingSquares(g2, 0.5);
    expect(s2).toHaveLength(2);
    const dir = (s: number[]) => Math.sign((s[2] - s[0]) * (s[3] - s[1]));
    expect(dir(segs[0])).not.toBe(dir(s2[0]));
  });

  it('NaN 이 있는 셀은 건너뛴다', () => {
    const g = makeGrid([-2, 2], [-2, 2], 21, 21, (x, y) => (x > 0 ? NaN : x * x + y * y));
    const segs = marchingSquares(g, 1);
    expect(segs.length).toBeGreaterThan(0);
    for (const s of segs) for (const c of s) expect(Number.isFinite(c)).toBe(true);
  });
});
