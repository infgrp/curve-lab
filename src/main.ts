import './style.css';
import { linearClosedForm } from './core/linear.ts';
import { App } from './ui/app.ts';
import type { Stage1Handle } from './ui/stage1.ts';
import type { Stage2Handle } from './ui/stage2.ts';
import type { Stage3Handle } from './ui/stage3.ts';

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) throw new Error('#app 요소가 없습니다');

const app = new App(root);

if (import.meta.env.DEV) {
  // dev 전용 점검 훅: ?stage=1|2|3|result [&seed=N] [&machine=…] [&h=…] [&reveal=1] [&rest=1]
  const q = new URLSearchParams(location.search);
  const stage = q.get('stage');
  if (stage) {
    if (q.has('seed')) app.settings = { ...app.settings, fixedSeed: Number(q.get('seed')) >>> 0 };
    const ss = app.newSession();
    app.session = ss;
    const h = <H>(): H => (app as unknown as { handle: H }).handle;
    const m = q.get('machine');
    if (stage === '1') {
      app.showStage1();
      if (m === 'small' || m === 'good' || m === 'tooBig') h<Stage1Handle>().runMachine(m, false);
    } else if (stage === '2') {
      ss.linear = linearClosedForm(ss.data.train);
      app.showStage2();
      if (q.has('rest')) h<Stage2Handle>().setNeuron(2, 1.15, 0.8);
      if (m) h<Stage2Handle>().runMachine(false);
    } else if (stage === '3') {
      app.showStage3();
      if (m) h<Stage3Handle>().trainNow(Number(q.get('h') ?? '8'));
      if (q.has('reveal')) h<Stage3Handle>().reveal();
    } else if (stage === 'result') {
      ss.linear = linearClosedForm(ss.data.train);
      app.showStage3();
      const s3 = h<Stage3Handle>();
      s3.trainNow(Number(q.get('h') ?? '8'));
      s3.reveal();
      root.querySelector<HTMLButtonElement>('[data-role="next"]')?.click(); // 실제 결과값으로 결과 화면
    }
    if (q.has('settings')) app.openSettings();
  }
  (window as unknown as { __app: App }).__app = app;
}
