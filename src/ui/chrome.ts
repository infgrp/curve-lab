/**
 * 화면 공통 조각: 배경 유리 구슬, 단계 머리글(단계 표시 + 제목 + 한 줄 안내 + '자세히'), 범례.
 * 수학은 없다.
 */
import { T } from './text.ko.ts';

export const TOTAL_STEPS = 3;

/** body 에 배경 구슬을 한 번만 넣는다 */
export function ensureBackground(): void {
  if (document.querySelector('.bg')) return;
  const bg = document.createElement('div');
  bg.className = 'bg';
  bg.setAttribute('aria-hidden', 'true');
  bg.innerHTML = '<i class="o1"></i><i class="o2"></i><i class="o3"></i><i class="o4"></i>';
  document.body.prepend(bg);
}

export function stageHeader(step: number, title: string, lead: string, guide: readonly string[]): string {
  return `
    <header class="stage-head">
      <div>
        <div class="titlebar"><span class="step-chip"><b>${step}</b> / ${TOTAL_STEPS} 단계</span><h1>${title}</h1></div>
        <p class="lead">${lead}</p>
      </div>
      <details class="more">
        <summary>${T.app.more} ▾</summary>
        <ol class="card">${guide.map((g) => `<li>${g}</li>`).join('')}</ol>
      </details>
    </header>`;
}

export type LegendItem = { kind: 'line' | 'dot' | 'dash' | 'diamond'; color: string; label: string };

export function legend(items: LegendItem[]): string {
  return `<div class="legend">${items
    .map((it) => {
      const style = it.kind === 'dash' ? `color:${it.color}` : `background:${it.color}`;
      const cls = it.kind === 'dot' ? 'dot' : it.kind === 'dash' ? 'dash' : it.kind === 'diamond' ? 'dot diamond' : '';
      return `<span><i class="${cls}" style="${style}"></i>${it.label}</span>`;
    })
    .join('')}</div>`;
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c);
}
