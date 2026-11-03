/**
 * 유한차분 기울기 검사기 (PROCEDURE 2-3, CLAUDE.md 규칙 3).
 *
 * 매개변수 벡터 p 와 손실 L(p), 해석적 기울기 g(p) 를 받아 중앙차분
 *   ∂L/∂pᵢ ≈ (L(p + h eᵢ) − L(p − h eᵢ)) / (2h),  h = 1e-6
 * 으로 수치 기울기를 구하고 성분별 상대오차의 최댓값을 돌려준다.
 * 상대오차 = |g − n| / max(|g| + |n|, floor). floor 는 두 값이 모두 0 근처일 때
 * 0/0 을 피하기 위한 바닥(기본 1e-8)이다.
 */

export interface GradCheckResult {
  /** 성분별 상대오차의 최댓값 */
  maxRelError: number;
  /** 그 최댓값이 난 성분 번호 */
  worstIndex: number;
  numeric: Float64Array;
  analytic: Float64Array;
}

export function gradCheck(
  params: Float64Array,
  loss: (p: Float64Array) => number,
  grad: (p: Float64Array) => Float64Array,
  h = 1e-6,
  floor = 1e-8,
): GradCheckResult {
  const p = Float64Array.from(params);
  const analytic = Float64Array.from(grad(p));
  const numeric = new Float64Array(p.length);
  let maxRelError = 0;
  let worstIndex = -1;
  for (let i = 0; i < p.length; i++) {
    const orig = p[i];
    p[i] = orig + h;
    const lp = loss(p);
    p[i] = orig - h;
    const lm = loss(p);
    p[i] = orig;
    numeric[i] = (lp - lm) / (2 * h);
    const rel = Math.abs(analytic[i] - numeric[i]) / Math.max(Math.abs(analytic[i]) + Math.abs(numeric[i]), floor);
    if (rel > maxRelError) {
      maxRelError = rel;
      worstIndex = i;
    }
  }
  return { maxRelError, worstIndex, numeric, analytic };
}
