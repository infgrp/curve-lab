/**
 * 방문자 기록 저장소 (SPEC 3단계). localStorage, 익명, 날짜별. 개인정보 없음.
 * 모든 localStorage 접근은 try/catch 로 감싸고, 실패해도 앱은 동작한다 (CLAUDE.md 규칙 5).
 */
import { type ScoreRecord, dayKey, sanitizeRecords } from '../core/scoreboard.ts';

const PREFIX = 'curve-lab:scores:';

export class ScoreStore {
  private readonly now: () => number;
  /** 저장소를 못 쓸 때의 메모리 보관 */
  private fallback: ScoreRecord[] = [];

  constructor(now: () => number = () => Date.now()) {
    this.now = now;
  }

  private key(): string {
    return PREFIX + dayKey(new Date(this.now()));
  }

  /** 오늘 기록 */
  today(): ScoreRecord[] {
    try {
      const raw = localStorage.getItem(this.key());
      return raw ? sanitizeRecords(JSON.parse(raw)) : [...this.fallback];
    } catch {
      return [...this.fallback];
    }
  }

  /** 기록 추가 후 오늘 기록 전체를 돌려준다 */
  add(record: Omit<ScoreRecord, 't'>): ScoreRecord[] {
    const r: ScoreRecord = { ...record, t: this.now() };
    const list = this.today();
    list.push(r);
    try {
      localStorage.setItem(this.key(), JSON.stringify(list));
    } catch {
      this.fallback = list;
    }
    return list;
  }

  /** 오늘 기록 삭제 (교사 설정) */
  clearToday(): void {
    this.fallback = [];
    try {
      localStorage.removeItem(this.key());
    } catch {
      /* 무시 */
    }
  }

  /** 저장된 모든 날짜 키 (교사 설정의 내보내기용) */
  days(): string[] {
    const out: string[] = [];
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith(PREFIX)) out.push(k.slice(PREFIX.length));
      }
    } catch {
      /* 무시 */
    }
    return out.sort();
  }

  forDay(day: string): ScoreRecord[] {
    try {
      const raw = localStorage.getItem(PREFIX + day);
      return raw ? sanitizeRecords(JSON.parse(raw)) : [];
    } catch {
      return [];
    }
  }
}
