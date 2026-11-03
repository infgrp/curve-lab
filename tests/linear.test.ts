import { describe, expect, it } from 'vitest';
import { generateDataset } from '../src/core/data.ts';
import {
  ETA_CHOICES,
  linearClosedForm,
  linearEtaMax,
  linearGdPath,
  linearGdStep,
  linearGrad,
  linearHessian,
  linearLambdaMax,
  linearLoss,
} from '../src/core/linear.ts';
import { gradCheck } from './helpers/gradcheck.ts';

const data = generateDataset(11).train;

describe('직선 모델', () => {
  it('(1) 기울기가 중앙차분 수치 기울기와 일치한다', () => {
    const loss = (p: Float64Array) => linearLoss(data, { a: p[0], b: p[1] });
    const grad = (p: Float64Array) => {
      const g = linearGrad(data, { a: p[0], b: p[1] });
      return Float64Array.of(g.a, g.b);
    };
    for (const p of [[0.3, -0.2], [-1.5, 2], [0, 0]]) {
      const r = gradCheck(Float64Array.from(p), loss, grad);
      expect(r.maxRelError).toBeLessThan(1e-6);
    }
    // 돌연변이 검사: 기울기 부호를 뒤집은 구현은 걸려야 한다
    const wrong = (p: Float64Array) => grad(p).map((v) => -v);
    expect(gradCheck(Float64Array.of(0.3, -0.2), loss, wrong).maxRelError).toBeGreaterThan(0.5);
  });

  it('닫힌 해에서 기울기가 0 이고, 그 손실이 주변보다 작다', () => {
    const p = linearClosedForm(data);
    const g = linearGrad(data, p);
    expect(Math.abs(g.a)).toBeLessThan(1e-12);
    expect(Math.abs(g.b)).toBeLessThan(1e-12);
    const L = linearLoss(data, p);
    expect(linearLoss(data, { a: p.a + 0.1, b: p.b })).toBeGreaterThan(L);
    expect(linearLoss(data, { a: p.a, b: p.b - 0.1 })).toBeGreaterThan(L);
  });

  it('(2) 적당한 η (0.6·η_max) 에서 경사하강법이 닫힌 해에 1e-6 이내로 수렴한다', () => {
    const star = linearClosedForm(data);
    const eta = ETA_CHOICES.good * linearEtaMax(data.x);
    let p = { a: 3, b: -3 };
    for (let i = 0; i < 2000; i++) p = linearGdStep(data, p, eta);
    expect(Math.abs(p.a - star.a)).toBeLessThan(1e-6);
    expect(Math.abs(p.b - star.b)).toBeLessThan(1e-6);
    // 작은 학습률(0.1·η_max)도 느리지만 수렴한다
    const etaSmall = ETA_CHOICES.small * linearEtaMax(data.x);
    let q = { a: 3, b: -3 };
    for (let i = 0; i < 20000; i++) q = linearGdStep(data, q, etaSmall);
    expect(Math.abs(q.a - star.a)).toBeLessThan(1e-6);
  });

  it('(3) 너무 큰 η (1.1·η_max) 에서는 손실이 증가한다(발산)', () => {
    const eta = ETA_CHOICES.tooBig * linearEtaMax(data.x);
    const star = linearClosedForm(data);
    const start = { a: star.a + 0.5, b: star.b + 0.5 };
    const path = linearGdPath(data, start, eta, 200);
    const L0 = linearLoss(data, start);
    const Lend = linearLoss(data, path[path.length - 1]);
    expect(Lend).toBeGreaterThan(L0 * 10);
    // 경로가 유한하지 않게 되어도 예외 없이 끝난다
    const far = linearGdPath(data, start, eta, 5000);
    expect(far.length).toBeGreaterThan(1);
  });

  it('(4) λ_max 가 2×2 고유값 공식과 일치한다', () => {
    const { h11, h12, h22 } = linearHessian(data.x);
    const lam = linearLambdaMax(data.x);
    // 특성다항식 λ² − tr·λ + det = 0 을 만족하고, 두 근 중 큰 것
    const tr = h11 + h22;
    const det = h11 * h22 - h12 * h12;
    expect(Math.abs(lam * lam - tr * lam + det)).toBeLessThan(1e-12);
    const other = tr - lam;
    expect(lam).toBeGreaterThanOrEqual(other);
    // 고유벡터 검증: (H − λI) v = 0 인 v = (h12, λ − h11)
    const v = [h12, lam - h11];
    const r1 = h11 * v[0] + h12 * v[1] - lam * v[0];
    const r2 = h12 * v[0] + h22 * v[1] - lam * v[1];
    expect(Math.abs(r1)).toBeLessThan(1e-12);
    expect(Math.abs(r2)).toBeLessThan(1e-12);
    // 손으로 계산할 수 있는 경우: x = [-1, 1] → H = [[2, 0],[0, 2]] → λ_max = 2
    expect(linearLambdaMax([-1, 1])).toBeCloseTo(2, 12);
    // x = [1, 1] → H = [[2, 2],[2, 2]] → λ = 4, 0
    expect(linearLambdaMax([1, 1])).toBeCloseTo(4, 12);
  });

  it('η_max 바로 아래에서는 수렴하고 바로 위에서는 발산한다 (경계가 정확함)', () => {
    const etaMax = linearEtaMax(data.x);
    const star = linearClosedForm(data);
    const start = { a: star.a + 1, b: star.b + 1 };
    const below = linearGdPath(data, start, 0.98 * etaMax, 3000);
    const above = linearGdPath(data, start, 1.02 * etaMax, 3000);
    const pb = below[below.length - 1];
    const pa = above[above.length - 1];
    expect(Math.abs(pb.a - star.a) + Math.abs(pb.b - star.b)).toBeLessThan(1e-3);
    expect(linearLoss(data, pa)).toBeGreaterThan(linearLoss(data, start));
  });
});
