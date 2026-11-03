/**
 * 과적합 재현 실험 (PROCEDURE 6-1). reference/overfit_experiment.py 를 src/core 코드로 다시 한다.
 *
 *   node --experimental-strip-types scripts/experiment-overfit.ts [--seeds 20] [--epochs 20000] [--n 12] [--sigma 0.25] [--json out.json]
 *
 * 시드마다: 데이터(generateDataset, SPEC 0절: 학습 n개, 검증 40개는 학습 x 범위 안) + 신경망 초기화 → Adam(lr 0.01) 으로
 * epochs 번 전체 배치 학습 → 학습/검증 MSE. 은닉 뉴런 1, 2, 4, 8, 16, 64 에 대해 평균과 중앙값을 표로 낸다.
 * 난수 생성기가 Python 과 달라 숫자는 같지 않다. "작음 → 적당 → 큼" 의 U자형(검증 MSE)이 유지되는지를 본다.
 */
import { writeFileSync } from 'node:fs';
import { generateDataset, mse } from '../src/core/data.ts';
import { Mlp, MlpTrainer } from '../src/core/mlp.ts';
import { Rng } from '../src/core/rng.ts';

interface Args {
  seeds: number;
  epochs: number;
  n: number;
  sigma: number;
  hidden: number[];
  json: string | null;
}

function parseArgs(argv: string[]): Args {
  const a: Args = { seeds: 20, epochs: 20000, n: 12, sigma: 0.25, hidden: [1, 2, 4, 8, 16, 64], json: null };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    const v = argv[i + 1];
    if (k === '--seeds') a.seeds = Number(v), i++;
    else if (k === '--epochs') a.epochs = Number(v), i++;
    else if (k === '--n') a.n = Number(v), i++;
    else if (k === '--sigma') a.sigma = Number(v), i++;
    else if (k === '--hidden') a.hidden = v.split(',').map(Number), i++;
    else if (k === '--json') a.json = v, i++;
  }
  return a;
}

export interface RunResult {
  H: number;
  seed: number;
  train: number;
  val: number;
}

export function runOne(H: number, seed: number, epochs: number, n: number, sigma: number): RunResult {
  const data = generateDataset(seed, { n, sigma });
  // 초기화 시드는 데이터 시드에서 파생 (같은 seed 면 같은 결과)
  const net = Mlp.init(H, new Rng((seed * 7919 + 1) >>> 0));
  const trainer = new MlpTrainer(net, data.train, 0.01);
  trainer.run(epochs);
  const predVal = net.predictInto(data.val.x, new Float64Array(data.val.x.length));
  return { H, seed, train: trainer.lastLoss, val: mse(predVal, data.val.y) };
}

const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;
const median = (xs: number[]) => {
  const s = [...xs].sort((p, q) => p - q);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export function summarize(results: RunResult[], hidden: number[]) {
  return hidden.map((H) => {
    const rs = results.filter((r) => r.H === H);
    return {
      H,
      trainMean: mean(rs.map((r) => r.train)),
      trainMedian: median(rs.map((r) => r.train)),
      valMean: mean(rs.map((r) => r.val)),
      valMedian: median(rs.map((r) => r.val)),
      valMax: Math.max(...rs.map((r) => r.val)),
    };
  });
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const t0 = Date.now();
  const results: RunResult[] = [];
  for (const H of args.hidden) {
    for (let s = 0; s < args.seeds; s++) results.push(runOne(H, s, args.epochs, args.n, args.sigma));
  }
  const rows = summarize(results, args.hidden);
  console.log(`# 과적합 재현: n=${args.n}, σ=${args.sigma} (σ²=${(args.sigma * args.sigma).toFixed(4)}), 시드 ${args.seeds}개, Adam lr=0.01, ${args.epochs} epoch, ${((Date.now() - t0) / 1000).toFixed(1)}초`);
  console.log('| 뉴런 H | 학습 MSE 평균 | 학습 중앙값 | 검증 MSE 평균 | 검증 중앙값 | 검증 최대 |');
  console.log('|---|---|---|---|---|---|');
  for (const r of rows) {
    console.log(`| ${r.H} | ${r.trainMean.toFixed(4)} | ${r.trainMedian.toFixed(4)} | ${r.valMean.toFixed(4)} | ${r.valMedian.toFixed(4)} | ${r.valMax.toFixed(3)} |`);
  }
  // U자형 판정: 2 → 8 → 64 에서 가운데가 가장 작은가 (평균·중앙값 각각)
  const pick = (H: number) => rows.find((r) => r.H === H);
  const [h2, h8, h64] = [pick(2), pick(8), pick(64)];
  if (h2 && h8 && h64) {
    const uMean = h8.valMean < h2.valMean && h8.valMean < h64.valMean;
    const uMedian = h8.valMedian < h2.valMedian && h8.valMedian < h64.valMedian;
    console.log(`\nU자형(2→8→64, 검증 MSE): 평균 ${uMean ? '유지' : '아님'}, 중앙값 ${uMedian ? '유지' : '아님'}`);
  }
  if (args.json) {
    writeFileSync(args.json, JSON.stringify({ args, rows, results }, null, 1));
    console.log(`JSON: ${args.json}`);
  }
}

main();
