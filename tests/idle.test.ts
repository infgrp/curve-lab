import { describe, expect, it } from 'vitest';
import { IdleTimer, type Scheduler } from '../src/ui/idle.ts';

/** 손으로 시간을 돌리는 가짜 스케줄러 */
function fakeScheduler() {
  let now = 0;
  let nextId = 1;
  const pending = new Map<number, { at: number; fn: () => void }>();
  const s: Scheduler = {
    schedule(fn, ms) {
      const id = nextId++;
      pending.set(id, { at: now + ms, fn });
      return id;
    },
    cancel(h) {
      pending.delete(h as number);
    },
  };
  const advance = (ms: number) => {
    const target = now + ms;
    // 예정 시각 순서대로 실행
    for (;;) {
      let best: [number, { at: number; fn: () => void }] | null = null;
      for (const e of pending) if (e[1].at <= target && (!best || e[1].at < best[1].at)) best = e;
      if (!best) break;
      pending.delete(best[0]);
      now = best[1].at;
      best[1].fn();
    }
    now = target;
  };
  return { s, advance, pendingCount: () => pending.size };
}

describe('IdleTimer', () => {
  it('입력이 없으면 timeout 뒤에 onIdle 이 한 번 불린다', () => {
    const f = fakeScheduler();
    let idle = 0;
    const t = new IdleTimer(120_000, () => idle++, f.s);
    t.start();
    f.advance(119_999);
    expect(idle).toBe(0);
    f.advance(1);
    expect(idle).toBe(1);
    f.advance(500_000);
    expect(idle).toBe(1); // 다시 start/touch 하기 전에는 반복되지 않는다
  });

  it('touch 하면 타이머가 처음부터 다시 잰다', () => {
    const f = fakeScheduler();
    let idle = 0;
    const t = new IdleTimer(1000, () => idle++, f.s);
    t.start();
    f.advance(900);
    t.touch();
    f.advance(900);
    expect(idle).toBe(0);
    f.advance(100);
    expect(idle).toBe(1);
    expect(f.pendingCount()).toBe(0);
  });

  it('stop 하면 불리지 않고, start 전의 touch 는 무시된다', () => {
    const f = fakeScheduler();
    let idle = 0;
    const t = new IdleTimer(1000, () => idle++, f.s);
    t.touch();
    expect(f.pendingCount()).toBe(0);
    t.start();
    t.stop();
    f.advance(5000);
    expect(idle).toBe(0);
    expect(t.isRunning).toBe(false);
    expect(() => new IdleTimer(0, () => {}, f.s)).toThrow();
  });
});
