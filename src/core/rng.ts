/**
 * 시드 고정 난수 (CLAUDE.md 규칙 4: 앱 전체에서 Math.random() 대신 이것만 쓴다).
 *
 * - mulberry32: 32비트 상태의 빠른 난수. 같은 시드면 같은 수열.
 * - Box–Muller 로 표준정규난수. 한 번에 두 개가 나오므로 하나는 보관했다가 다음에 준다.
 */

/** 32비트 정수 시드 → [0, 1) 균등난수 생성 함수 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Rng {
  readonly seed: number;
  private readonly next01: () => number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.seed = seed;
    this.next01 = mulberry32(seed);
  }

  /** [0, 1) */
  uniform01(): number {
    return this.next01();
  }

  /** [lo, hi) */
  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next01();
  }

  /** 정수 [0, n) */
  int(n: number): number {
    return Math.floor(this.next01() * n);
  }

  /** N(mean, sd²) — Box–Muller */
  normal(mean = 0, sd = 1): number {
    if (this.spare !== null) {
      const z = this.spare;
      this.spare = null;
      return mean + sd * z;
    }
    let u = 0;
    while (u === 0) u = this.next01(); // log(0) 방지
    const v = this.next01();
    const r = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * v;
    this.spare = r * Math.sin(theta);
    return mean + sd * r * Math.cos(theta);
  }

  /** 이 난수기에서 파생된 새 시드 (하위 난수기를 만들 때) */
  nextSeed(): number {
    return Math.floor(this.next01() * 4294967296) >>> 0;
  }
}

/**
 * 세션 시드 만들기. core 는 시계를 직접 읽지 않으므로 호출자가 `now`(ms) 와
 * 선택적으로 엔트로피 정수(예: crypto.getRandomValues 로 얻은 값)를 넘긴다.
 * 교사 설정에서 시드를 고정하면 이 함수를 쓰지 않는다.
 */
export function seedFrom(now: number, entropy = 0): number {
  const lo = Math.floor(now) >>> 0;
  const hi = Math.floor(now / 4294967296) >>> 0;
  return (lo ^ Math.imul(hi, 0x9e3779b1) ^ (entropy >>> 0)) >>> 0;
}
