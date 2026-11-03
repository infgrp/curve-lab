import './style.css';

// Phase 1: 빈 화면에 제목만. 단계 화면은 src/ui/ 에서 Phase 3 이후에 붙인다.
const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('#app 요소가 없습니다');

const title = document.createElement('h1');
title.className = 'title';
title.textContent = '곡선 찾기 실험실';
app.replaceChildren(title);
