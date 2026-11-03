/**
 * 교사 설정 (SPEC 4절): 시드 고정, σ, 학습 점 수. 저장은 ui 가 하고 여기는 기본값·검사만.
 */

export interface Settings {
  /** null 이면 세션마다 새 시드 */
  fixedSeed: number | null;
  sigma: number;
  n: number;
  /** 무입력 초기화까지의 초 */
  idleSeconds: number;
}

export const DEFAULT_SETTINGS: Settings = { fixedSeed: null, sigma: 0.25, n: 12, idleSeconds: 120 };
export const SETTINGS_LIMITS = { sigma: [0, 1] as const, n: [4, 40] as const, idleSeconds: [20, 3600] as const };

const clamp = (v: number, [lo, hi]: readonly [number, number]) => Math.min(hi, Math.max(lo, v));

/** 어떤 입력이 와도 올바른 설정을 돌려준다 (저장소가 깨져 있어도 앱이 뜬다) */
export function sanitizeSettings(raw: unknown): Settings {
  const s = { ...DEFAULT_SETTINGS };
  if (!raw || typeof raw !== 'object') return s;
  const o = raw as Record<string, unknown>;
  const seed = o['fixedSeed'];
  if (typeof seed === 'number' && Number.isInteger(seed) && seed >= 0 && seed < 2 ** 32) s.fixedSeed = seed;
  if (typeof seed === 'string' && /^\d+$/.test(seed)) s.fixedSeed = Number(seed) >>> 0;
  const sigma = Number(o['sigma']);
  if (Number.isFinite(sigma)) s.sigma = clamp(sigma, SETTINGS_LIMITS.sigma);
  const n = Number(o['n']);
  if (Number.isInteger(n)) s.n = clamp(n, SETTINGS_LIMITS.n);
  const idle = Number(o['idleSeconds']);
  if (Number.isFinite(idle)) s.idleSeconds = clamp(Math.round(idle), SETTINGS_LIMITS.idleSeconds);
  return s;
}
