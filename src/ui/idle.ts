/**
 * 무입력 타이머 (SPEC 4절: 120초 동안 입력이 없으면 시작 화면으로). DOM 과 시계를 직접 쓰지 않고
 * 스케줄 함수를 주입받아 단위 테스트할 수 있게 한다 (PROCEDURE Phase 7).
 */

export interface Scheduler {
  /** ms 뒤에 fn 을 부르는 예약. 취소용 핸들을 돌려준다 */
  schedule(fn: () => void, ms: number): unknown;
  cancel(handle: unknown): void;
}

export const realScheduler: Scheduler = {
  schedule: (fn, ms) => setTimeout(fn, ms),
  cancel: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export class IdleTimer {
  readonly timeoutMs: number;
  private readonly onIdle: () => void;
  private readonly sched: Scheduler;
  private handle: unknown = null;
  private running = false;

  constructor(timeoutMs: number, onIdle: () => void, scheduler: Scheduler = realScheduler) {
    if (!(timeoutMs > 0)) throw new Error('timeoutMs 는 양수여야 합니다');
    this.timeoutMs = timeoutMs;
    this.onIdle = onIdle;
    this.sched = scheduler;
  }

  get isRunning(): boolean {
    return this.running;
  }

  /** 감시 시작 (이미 돌고 있으면 타이머만 다시 잰다) */
  start(): void {
    this.running = true;
    this.touch();
  }

  /** 입력이 있었다: 타이머를 처음부터 다시 잰다 */
  touch(): void {
    if (!this.running) return;
    if (this.handle !== null) this.sched.cancel(this.handle);
    this.handle = this.sched.schedule(() => {
      this.handle = null;
      if (this.running) this.onIdle();
    }, this.timeoutMs);
  }

  stop(): void {
    this.running = false;
    if (this.handle !== null) this.sched.cancel(this.handle);
    this.handle = null;
  }
}

/** document 의 입력 이벤트를 타이머에 잇는다. 해제 함수를 돌려준다. */
export function watchUserInput(target: EventTarget, timer: IdleTimer): () => void {
  const events = ['pointerdown', 'pointermove', 'pointerup', 'keydown', 'wheel', 'touchstart'] as const;
  const handler = () => timer.touch();
  for (const e of events) target.addEventListener(e, handler, { passive: true });
  return () => {
    for (const e of events) target.removeEventListener(e, handler);
  };
}
