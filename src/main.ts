import './style.css';
import { generateDataset } from './core/data.ts';
import { seedFrom } from './core/rng.ts';
import { mountStage1 } from './ui/stage1.ts';

// Phase 4 임시 연결: 1단계 화면을 바로 띄운다. 시작 화면·단계 이동·무입력 초기화는 Phase 7(app.ts) 에서.
const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('#app 요소가 없습니다');

const entropy = new Uint32Array(1);
crypto.getRandomValues(entropy); // Math.random 대신 (CLAUDE.md 규칙 4)
const seed = seedFrom(Date.now(), entropy[0]);
const data = generateDataset(seed);

const stage1 = mountStage1(app, data, {
  onNext: (r) => console.log('1단계 결과', r),
});

if (import.meta.env.DEV) {
  // dev 전용 점검 훅: ?machine=small|good|tooBig 이면 애니메이션 없이 경사하강을 끝까지 돌린 상태로 연다
  (window as unknown as { __stage1: typeof stage1 }).__stage1 = stage1;
  const m = new URLSearchParams(location.search).get('machine');
  if (m === 'small' || m === 'good' || m === 'tooBig') stage1.runMachine(m, false);
}
