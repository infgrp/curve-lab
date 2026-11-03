/**
 * 등고선 계산 (marching squares). 그리기와 무관한 순수 수학이라 core 에 둔다.
 *
 * 격자 Grid: xs (nx개), ys (ny개), 값 v 는 행 우선 v[j·nx + i] = f(xs[i], ys[j]).
 * marchingSquares(grid, level) 은 f = level 인 선분들을 수학 좌표로 돌려준다.
 */

export interface Grid {
  readonly xs: Float64Array;
  readonly ys: Float64Array;
  /** 길이 nx·ny, v[j·nx + i] */
  readonly v: Float64Array;
}

/** 선분 [x1, y1, x2, y2] */
export type Segment = [number, number, number, number];

export function makeGrid(
  xRange: readonly [number, number],
  yRange: readonly [number, number],
  nx: number,
  ny: number,
  f: (x: number, y: number) => number,
): Grid {
  if (nx < 2 || ny < 2) throw new Error('격자는 각 방향 2칸 이상이어야 합니다');
  const xs = new Float64Array(nx);
  const ys = new Float64Array(ny);
  for (let i = 0; i < nx; i++) xs[i] = xRange[0] + ((xRange[1] - xRange[0]) * i) / (nx - 1);
  for (let j = 0; j < ny; j++) ys[j] = yRange[0] + ((yRange[1] - yRange[0]) * j) / (ny - 1);
  const v = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) v[j * nx + i] = f(xs[i], ys[j]);
  return { xs, ys, v };
}

export function gridMinMax(grid: Grid): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const val of grid.v) {
    if (!Number.isFinite(val)) continue;
    if (val < min) min = val;
    if (val > max) max = val;
  }
  return { min, max };
}

/** lo 와 hi 사이(양 끝 제외)의 로그 간격 등고선 수준 k개. lo > 0 이어야 한다. */
export function logLevels(lo: number, hi: number, k: number): number[] {
  if (!(lo > 0) || !(hi > lo) || k < 1) return [];
  const out: number[] = [];
  const a = Math.log(lo);
  const b = Math.log(hi);
  for (let i = 1; i <= k; i++) out.push(Math.exp(a + ((b - a) * i) / (k + 1)));
  return out;
}

/** 두 꼭짓점 값 사이에서 level 이 되는 위치 (선형 보간) */
function lerpPos(p0: number, p1: number, v0: number, v1: number, level: number): number {
  const d = v1 - v0;
  if (d === 0) return (p0 + p1) / 2;
  return p0 + ((level - v0) / d) * (p1 - p0);
}

/**
 * 셀 하나에서 등고선 선분을 뽑는다. 꼭짓점 번호: 0=(x0,y0) 1=(x1,y0) 2=(x1,y1) 3=(x0,y1).
 * 변 번호: 0=아래(0-1) 1=오른쪽(1-2) 2=위(3-2) 3=왼쪽(0-3). 안장(5, 10)은 중심값으로 가른다.
 */
const EDGE_TABLE: ReadonlyArray<ReadonlyArray<[number, number]>> = [
  [], // 0
  [[3, 0]], // 1: v0
  [[0, 1]], // 2: v1
  [[3, 1]], // 3: v0 v1
  [[1, 2]], // 4: v2
  [], // 5: v0 v2 (안장, 별도 처리)
  [[0, 2]], // 6: v1 v2
  [[3, 2]], // 7: v0 v1 v2
  [[2, 3]], // 8: v3
  [[0, 2]], // 9: v0 v3
  [], // 10: v1 v3 (안장)
  [[1, 2]], // 11: v0 v1 v3
  [[1, 3]], // 12: v2 v3
  [[0, 1]], // 13: v0 v2 v3
  [[3, 0]], // 14: v1 v2 v3
  [], // 15
];

export function marchingSquares(grid: Grid, level: number): Segment[] {
  const { xs, ys, v } = grid;
  const nx = xs.length;
  const ny = ys.length;
  const out: Segment[] = [];
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const v0 = v[j * nx + i];
      const v1 = v[j * nx + i + 1];
      const v2 = v[(j + 1) * nx + i + 1];
      const v3 = v[(j + 1) * nx + i];
      if (!Number.isFinite(v0) || !Number.isFinite(v1) || !Number.isFinite(v2) || !Number.isFinite(v3)) continue;
      // '높음' = 수준보다 엄격히 큼. 꼭짓점 값이 수준과 정확히 같으면 '낮음'으로 두어, 이웃 셀들이 그 꼭짓점에서
      // 같은 점을 중복으로 만들지 않게 한다 (그래도 생기는 길이 0 선분은 아래에서 버린다).
      const code = (v0 > level ? 1 : 0) | (v1 > level ? 2 : 0) | (v2 > level ? 4 : 0) | (v3 > level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      const x0 = xs[i];
      const x1 = xs[i + 1];
      const y0 = ys[j];
      const y1 = ys[j + 1];
      const edgePoint = (e: number): [number, number] => {
        switch (e) {
          case 0:
            return [lerpPos(x0, x1, v0, v1, level), y0];
          case 1:
            return [x1, lerpPos(y0, y1, v1, v2, level)];
          case 2:
            return [lerpPos(x0, x1, v3, v2, level), y1];
          default:
            return [x0, lerpPos(y0, y1, v0, v3, level)];
        }
      };
      let pairs: ReadonlyArray<[number, number]>;
      if (code === 5 || code === 10) {
        const center = (v0 + v1 + v2 + v3) / 4;
        const centerHigh = center > level;
        // 5: v0, v2 높음. 중심도 높으면 두 높은 꼭짓점이 이어진다 → (3,2),(0,1); 아니면 (3,0),(1,2)
        if (code === 5) pairs = centerHigh ? [[3, 2], [0, 1]] : [[3, 0], [1, 2]];
        else pairs = centerHigh ? [[3, 0], [1, 2]] : [[3, 2], [0, 1]];
      } else {
        pairs = EDGE_TABLE[code];
      }
      for (const [ea, eb] of pairs) {
        const [ax, ay] = edgePoint(ea);
        const [bx, by] = edgePoint(eb);
        // 꼭짓점을 (부동소수 오차 안에서) 정확히 지나면 두 끝점이 같은 퇴화 선분이 나오므로 버린다
        if (Math.abs(ax - bx) + Math.abs(ay - by) < 1e-9 * (Math.abs(x1 - x0) + Math.abs(y1 - y0))) continue;
        out.push([ax, ay, bx, by]);
      }
    }
  }
  return out;
}
