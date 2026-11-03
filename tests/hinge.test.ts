import { describe, expect, it } from 'vitest';
import { generateDataset } from '../src/core/data.ts';
import {
  dormantNeurons,
  fromReluNetwork,
  hingeFromVector,
  hingeGdStep,
  hingeGrad,
  hingeInitFromLinear,
  hingeLoss,
  hingePredict,
  hingeToVector,
  reluNetworkPredict,
} from '../src/core/hinge.ts';
import { linearClosedForm, linearLoss } from '../src/core/linear.ts';
import { Rng } from '../src/core/rng.ts';
import { gradCheck } from './helpers/gradcheck.ts';

const data = generateDataset(21).train;

describe('꺾은선 모델', () => {
  it('초기화: 1단계의 (a, b) 를 잇고 t = −0.5, 0, 0.5, s = 0 이라 손실이 직선과 같다', () => {
    const lin = linearClosedForm(data);
    const p = hingeInitFromLinear(lin);
    expect(Array.from(p.t)).toEqual([-0.5, 0, 0.5]);
    expect(Array.from(p.s)).toEqual([0, 0, 0]);
    expect(hingeLoss(data, p)).toBeCloseTo(linearLoss(data, lin), 14);
  });

  it('(1) 기울기 유한차분 검사 — 꺾이는 점에서 떨어진 x 만 사용', () => {
    // max(0, x − t) 는 x = t 에서 미분 불가능하고, 중앙차분(h = 1e-6)은 [t − h, t + h] 를 가로지르면
    // 틀린 값을 내므로 모든 |xᵢ − tₖ| 가 1e-3 이상이 되도록 t 를 고른다.
    const K = 3;
    const rng = new Rng(3);
    for (let trial = 0; trial < 5; trial++) {
      const p0 = {
        a: rng.uniform(-1, 1),
        b: rng.uniform(-1, 1),
        s: Float64Array.from({ length: K }, () => rng.uniform(-2, 2)),
        t: Float64Array.from({ length: K }, () => rng.uniform(-0.9, 0.9)),
      };
      for (let k = 0; k < K; k++) {
        while (Array.from(data.x).some((x) => Math.abs(x - p0.t[k]) < 1e-3)) p0.t[k] += 2e-3;
      }
      const loss = (v: Float64Array) => hingeLoss(data, hingeFromVector(v, K));
      const grad = (v: Float64Array) => {
        const g = hingeGrad(data, hingeFromVector(v, K));
        return hingeToVector(g);
      };
      const r = gradCheck(hingeToVector(p0), loss, grad);
      expect(r.maxRelError).toBeLessThan(1e-6);
    }
    // 돌연변이: t 의 기울기 부호를 뒤집으면 걸린다
    const p = hingeInitFromLinear({ a: 0.2, b: 0.1 });
    p.s.set([1, -1, 0.5]);
    const loss = (v: Float64Array) => hingeLoss(data, hingeFromVector(v, K));
    const wrong = (v: Float64Array) => {
      const g = hingeToVector(hingeGrad(data, hingeFromVector(v, K)));
      for (let k = 0; k < K; k++) g[2 + K + k] = -g[2 + K + k];
      return g;
    };
    expect(gradCheck(hingeToVector(p), loss, wrong).maxRelError).toBeGreaterThan(0.5);
  });

  it('(2) 임의의 ReLU 신경망과 변환한 꺾은선 모델이 무작위 x 1000개에서 1e-12 이내로 같다', () => {
    const rng = new Rng(77);
    for (let trial = 0; trial < 20; trial++) {
      const K = 5;
      const net = {
        w: Array.from({ length: K }, () => rng.uniform(-3, 3)),
        beta: Array.from({ length: K }, () => rng.uniform(-2, 2)),
        v: Array.from({ length: K }, () => rng.uniform(-2, 2)),
        c: rng.uniform(-1, 1),
      };
      // w < 0 과 w = 0 경우도 반드시 포함
      net.w[0] = -Math.abs(net.w[0]) - 0.1;
      net.w[1] = 0;
      const p = fromReluNetwork(net);
      expect(p.s.length).toBe(K - 1); // w = 0 인 뉴런 하나는 상수로 흡수
      for (let i = 0; i < 1000; i++) {
        const x = rng.uniform(-3, 3);
        expect(Math.abs(hingePredict(p, x) - reluNetworkPredict(net, x))).toBeLessThan(1e-12);
      }
    }
  });

  it('(3) 경사하강법으로 학습하면 손실이 1단계 최적 직선의 손실보다 작거나 같다', () => {
    const lin = linearClosedForm(data);
    const L0 = linearLoss(data, lin);
    let p = hingeInitFromLinear(lin);
    let best = hingeLoss(data, p);
    for (let i = 0; i < 3000; i++) {
      p = hingeGdStep(data, p, 0.05);
      best = Math.min(best, hingeLoss(data, p));
    }
    const L = hingeLoss(data, p);
    expect(L).toBeLessThanOrEqual(L0 + 1e-12);
    expect(L).toBeLessThan(0.8 * L0); // 사인 곡선이라 꺾은선이 직선보다 뚜렷이 낫다
    expect(best).toBeLessThanOrEqual(L0 + 1e-12);
  });

  it('쉬고 있는 뉴런: 꺾이는 점이 모든 학습 점의 오른쪽이면 기울기가 0 이고 움직이지 않는다', () => {
    const p = hingeInitFromLinear({ a: 0, b: 0 });
    p.t[2] = 1.5; // 학습 점은 [-1, 1] 안
    p.s[2] = 0.7;
    expect(dormantNeurons(p, data.x)).toEqual([false, false, true]);
    const g = hingeGrad(data, p);
    expect(g.s[2]).toBe(0);
    expect(g.t[2]).toBe(0);
    let q = p;
    for (let i = 0; i < 100; i++) q = hingeGdStep(data, q, 0.05);
    expect(q.t[2]).toBe(1.5);
    expect(q.s[2]).toBe(0.7);
    expect(q.s[0]).not.toBe(p.s[0]);
  });

  it('벡터 변환이 왕복한다', () => {
    const p = hingeInitFromLinear({ a: 1, b: 2 });
    p.s.set([3, 4, 5]);
    const v = hingeToVector(p);
    expect(Array.from(v)).toEqual([1, 2, 3, 4, 5, -0.5, 0, 0.5]);
    const q = hingeFromVector(v, 3);
    expect(q).toEqual(p);
    expect(() => hingeFromVector(v, 2)).toThrow();
  });
});
