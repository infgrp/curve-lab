import { describe, expect, it } from 'vitest';
import { generateDataset, mse } from '../src/core/data.ts';
import { Mlp, MlpTrainer } from '../src/core/mlp.ts';
import { Rng } from '../src/core/rng.ts';
import { gradCheck } from './helpers/gradcheck.ts';

describe('신경망 1 → H → 1', () => {
  it('초기화: 길이 3H+1, b₁ = b₂ = 0, W 의 분산이 SPEC 과 맞다 (큰 H 로 통계 확인)', () => {
    const H = 4000;
    const net = Mlp.init(H, new Rng(1));
    expect(net.params.length).toBe(3 * H + 1);
    expect(Array.from(net.b1).every((v) => v === 0)).toBe(true);
    expect(net.b2).toBe(0);
    const varOf = (a: Float64Array) => a.reduce((s, v) => s + v * v, 0) / a.length;
    // Var(W₁) = 2, Var(W₂) = 2/H. 표본분산의 표준오차 ≈ Var·√(2/H) → 4배
    expect(Math.abs(varOf(net.W1) - 2)).toBeLessThan(4 * 2 * Math.sqrt(2 / H));
    expect(Math.abs(varOf(net.W2) - 2 / H)).toBeLessThan((4 * 2 / H) * Math.sqrt(2 / H));
  });

  it('(1) 역전파 기울기가 모든 매개변수에서 유한차분과 일치한다', () => {
    const data = generateDataset(5, { n: 8 }).train;
    const rng = new Rng(9);
    let checked = 0;
    for (let trial = 0; trial < 20 && checked < 5; trial++) {
      const net = Mlp.init(4, rng);
      for (let j = 0; j < 4; j++) net.b1[j] = rng.uniform(-0.5, 0.5);
      net.params[3 * 4] = rng.uniform(-0.5, 0.5);
      // ReLU 는 z = 0 에서 미분 불가능: 어떤 (점, 뉴런) 의 |z| 가 1e-4 보다 작으면 이 초기값은 건너뛴다
      let nearKink = false;
      for (const x of data.x) for (let j = 0; j < 4; j++) if (Math.abs(net.W1[j] * x + net.b1[j]) < 1e-4) nearKink = true;
      if (nearKink) continue;
      const probe = new Mlp(4);
      const loss = (p: Float64Array) => {
        probe.params.set(p);
        return probe.loss(data);
      };
      const grad = (p: Float64Array) => {
        probe.params.set(p);
        probe.lossAndGrad(data);
        return Float64Array.from(probe.grad);
      };
      const r = gradCheck(net.params, loss, grad);
      expect(r.maxRelError).toBeLessThan(1e-6);
      checked++;
    }
    expect(checked).toBeGreaterThanOrEqual(3);
  });

  it('lossAndGrad 의 손실이 loss() 와 같고, 돌연변이(W₂ 기울기 부호 반전)는 검사에 걸린다', () => {
    const data = generateDataset(5, { n: 8 }).train;
    const net = Mlp.init(3, new Rng(2));
    expect(net.lossAndGrad(data)).toBeCloseTo(net.loss(data), 14);
    const probe = new Mlp(3);
    const loss = (p: Float64Array) => {
      probe.params.set(p);
      return probe.loss(data);
    };
    const wrong = (p: Float64Array) => {
      probe.params.set(p);
      probe.lossAndGrad(data);
      const g = Float64Array.from(probe.grad);
      for (let j = 6; j < 9; j++) g[j] = -g[j];
      return g;
    };
    expect(gradCheck(net.params, loss, wrong).maxRelError).toBeGreaterThan(0.5);
  });

  it('(2) 시드가 같으면 학습 결과가 비트 단위로 같다', () => {
    const data = generateDataset(3).train;
    const run = () => {
      const tr = new MlpTrainer(Mlp.init(8, new Rng(42)), data);
      tr.run(300);
      return tr;
    };
    const a = run();
    const b = run();
    expect(Array.from(a.net.params)).toEqual(Array.from(b.net.params));
    expect(a.lastLoss).toBe(b.lastLoss);
    const c = new MlpTrainer(Mlp.init(8, new Rng(43)), data);
    c.run(300);
    expect(Array.from(c.net.params)).not.toEqual(Array.from(a.net.params));
  });

  it('(3) H = 64, 20000 epoch 에서 학습 MSE < 0.02', () => {
    const data = generateDataset(1).train;
    const tr = new MlpTrainer(Mlp.init(64, new Rng(7)), data);
    tr.run(20000);
    expect(tr.epoch).toBe(20000);
    expect(tr.lastLoss).toBeLessThan(0.02);
    const pred = tr.net.predictInto(data.x, new Float64Array(data.x.length));
    expect(mse(pred, data.y)).toBeCloseTo(tr.net.loss(data), 12);
  });

  it('학습 중 메모리를 새로 할당하지 않는다 (버퍼가 같은 객체로 유지)', () => {
    const data = generateDataset(1).train;
    const net = Mlp.init(8, new Rng(1));
    net.lossAndGrad(data);
    const g1 = net.grad;
    const p1 = net.params;
    net.lossAndGrad(data);
    expect(net.grad).toBe(g1);
    expect(net.params).toBe(p1);
  });
});
