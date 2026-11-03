import { describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS, sanitizeSettings } from '../src/core/settings.ts';

describe('sanitizeSettings', () => {
  it('깨진 입력은 기본값, 범위 밖은 잘라낸다', () => {
    expect(sanitizeSettings(null)).toEqual(DEFAULT_SETTINGS);
    expect(sanitizeSettings('x')).toEqual(DEFAULT_SETTINGS);
    const s = sanitizeSettings({ fixedSeed: 42, sigma: 5, n: 2, idleSeconds: 1 });
    expect(s).toEqual({ fixedSeed: 42, sigma: 1, n: 4, idleSeconds: 20 });
  });
  it('시드는 정수 문자열도 받고, 음수·소수는 무시한다', () => {
    expect(sanitizeSettings({ fixedSeed: '123' }).fixedSeed).toBe(123);
    expect(sanitizeSettings({ fixedSeed: -1 }).fixedSeed).toBeNull();
    expect(sanitizeSettings({ fixedSeed: 1.5 }).fixedSeed).toBeNull();
    expect(sanitizeSettings({ sigma: 0.1, n: 20.7 })).toEqual({ ...DEFAULT_SETTINGS, sigma: 0.1 });
  });
});
