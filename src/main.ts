import './style.css';
import { generateDataset } from './core/data.ts';
import { linearClosedForm } from './core/linear.ts';
import { seedFrom } from './core/rng.ts';
import { ensureBackground } from './ui/chrome.ts';
import { mountStage1 } from './ui/stage1.ts';
import { mountStage2 } from './ui/stage2.ts';
import { mountStage3 } from './ui/stage3.ts';

// 임시 흐름(Phase 7 전): 1단계 → 2단계. 시작 화면·3단계·무입력 초기화는 Phase 6~7(app.ts) 에서.
const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('#app 요소가 없습니다');
ensureBackground();

const entropy = new Uint32Array(1);
crypto.getRandomValues(entropy); // Math.random 대신 (CLAUDE.md 규칙 4)
const query = new URLSearchParams(location.search);
const seed = query.has('seed') ? Number(query.get('seed')) >>> 0 : seedFrom(Date.now(), entropy[0]);
const data = generateDataset(seed);

function openStage1(): void {
  const stage1 = mountStage1(app!, data, { onNext: (r) => openStage2(r) });
  devHook(1, (m) => {
    if (m === 'small' || m === 'good' || m === 'tooBig') stage1.runMachine(m, false);
  });
}

function openStage2(linear: { a: number; b: number }): void {
  const stage2 = mountStage2(app!, data, linear, { onNext: () => openStage3() });
  devHook(2, (m) => {
    if (query.has('rest')) stage2.setNeuron(2, 1.15, 0.8); // 뉴런 3 을 자료 오른쪽 너머로 → 쉬는 뉴런 점검
    if (m) stage2.runMachine(false);
  });
}

function openStage3(): void {
  const stage3 = mountStage3(app!, data, { onNext: (r) => console.log('3단계 결과', r) });
  devHook(3, (m) => {
    const h = Number(query.get('h') ?? '8');
    if (m) stage3.trainNow(h);
    if (query.has('reveal')) stage3.reveal();
  });
}

/** dev 전용 점검 훅: ?stage=N&machine=… 로 특정 상태를 애니메이션 없이 연다 */
function devHook(stage: number, run: (machine: string | null) => void): void {
  if (!import.meta.env.DEV) return;
  const wanted = Number(query.get('stage') ?? '1');
  if (wanted === stage) run(query.get('machine'));
}

const wantedStage = import.meta.env.DEV ? Number(query.get('stage')) : 0;
if (wantedStage === 2) openStage2(linearClosedForm(data.train));
else if (wantedStage === 3) openStage3();
else openStage1();
