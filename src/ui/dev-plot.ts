/**
 * plot.ts 확인용 임시 페이지 (dev 전용, `npm run dev` 후 /dev-plot.html).
 * 왼쪽: 참 함수 곡선, 학습 점(원), 검증 점(다이아몬드), 직선과 잔차.
 * 오른쪽: (a, b) 평면의 직선 손실 등고선(로그 간격)과 닫힌 해·경사하강 경로.
 */
import '../style.css';
import { logLevels, makeGrid } from '../core/contour.ts';
import { TRUE_FN, generateDataset } from '../core/data.ts';
import { ETA_CHOICES, linearClosedForm, linearEtaMax, linearGdPath, linearLoss } from '../core/linear.ts';
import { COLORS, Plot } from './plot.ts';

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('#app 없음');
app.innerHTML = `
  <style>
    #app { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; width: 100vw; height: 100vh; padding: 24px; text-align: left; }
    .panel { display: grid; grid-template-rows: auto 1fr; gap: 8px; min-height: 0; }
    .panel h2 { margin: 0; font-size: 28px; }
    .panel canvas { width: 100%; height: 100%; background: #fff; border-radius: 16px; box-shadow: 0 2px 12px rgba(21,34,58,.08); display: block; }
  </style>
  <section class="panel"><h2>곡선·점·잔차 (시드 1)</h2><canvas id="c1"></canvas></section>
  <section class="panel"><h2>손실 등고선 (a, b)</h2><canvas id="c2"></canvas></section>
`;

const data = generateDataset(1);
const star = linearClosedForm(data.train);
const line = { a: star.a * 0.4, b: star.b + 0.6 }; // 일부러 최적이 아닌 직선

const c1 = new Plot(document.querySelector<HTMLCanvasElement>('#c1')!, {
  xRange: [-1.2, 1.2],
  yRange: [-2, 2],
  xLabel: 'x',
  yLabel: 'y',
});
c1.onResize(() => {
  c1.clear();
  c1.drawAxes();
  c1.drawCurve(TRUE_FN, { color: 'rgba(21,34,58,.35)', width: 2, dash: [8, 6] });
  c1.drawCurve((x) => line.a * x + line.b, { color: COLORS.train, width: 3 });
  c1.drawResiduals(data.train.x, data.train.y, (x) => line.a * x + line.b);
  c1.drawPoints(data.val.x, data.val.y, { shape: 'diamond', color: COLORS.val, radius: 6 });
  c1.drawPoints(data.train.x, data.train.y, { shape: 'circle', color: COLORS.train, radius: 8 });
  c1.drawText(`손실 ${linearLoss(data.train, line).toFixed(3)}`, -1.1, 1.8);
});

const aRange: [number, number] = [-2, 3];
const bRange: [number, number] = [-2, 2];
const grid = makeGrid(aRange, bRange, 121, 121, (a, b) => linearLoss(data.train, { a, b }));
const minLoss = linearLoss(data.train, star);
let maxLoss = 0;
for (const v of grid.v) maxLoss = Math.max(maxLoss, v);
const levels = logLevels(minLoss * 1.05, maxLoss, 12);
const path = linearGdPath(data.train, { a: -1.5, b: 1.5 }, ETA_CHOICES.good * linearEtaMax(data.train.x), 25);

const c2 = new Plot(document.querySelector<HTMLCanvasElement>('#c2')!, {
  xRange: aRange,
  yRange: bRange,
  xLabel: 'a (기울기)',
  yLabel: 'b (절편)',
});
c2.onResize(() => {
  c2.clear();
  c2.drawAxes();
  c2.drawContours(grid, levels);
  c2.drawPath(path.map((p) => ({ x: p.a, y: p.b })));
  c2.drawMarker(star.a, star.b, { color: COLORS.val, radius: 8 });
  c2.drawMarker(line.a, line.b);
  c2.drawText('최솟점', star.a, star.b, { dx: 14, dy: -10, color: COLORS.val });
});
