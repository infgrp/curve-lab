import { describe, expect, it } from 'vitest';
import { betterThanFraction, binCounts, binIndex, dayKey, meansByH, sanitizeRecords, toCsv } from '../src/core/scoreboard.ts';

describe('scoreboard core', () => {
  it('dayKey 는 로컬 날짜 YYYY-MM-DD', () => {
    expect(dayKey(new Date(2026, 10, 3, 9, 30))).toBe('2026-11-03');
    expect(dayKey(new Date(2026, 0, 9))).toBe('2026-01-09');
  });

  it('binCounts / binIndex: 범위 밖 처리와 마지막 칸', () => {
    const c = binCounts([0, 0.05, 0.1, 0.39, 0.4, 5, -1, NaN], 0, 0.4, 4);
    expect(c).toEqual([2, 1, 0, 3]); // 0,0.05 → 0칸; 0.1 → 1칸; 0.39,0.4,5 → 마지막 칸; -1, NaN 제외
    expect(binIndex(0.1, 0, 0.4, 4)).toBe(1);
    expect(binIndex(9, 0, 0.4, 4)).toBe(3);
    expect(binIndex(-0.1, 0, 0.4, 4)).toBe(-1);
  });

  it('meansByH 는 H 오름차순 평균, betterThanFraction 은 더 나쁜 기록의 비율', () => {
    const rs = [
      { H: 64, val: 0.2, train: 0, t: 1 },
      { H: 8, val: 0.1, train: 0, t: 2 },
      { H: 8, val: 0.3, train: 0, t: 3 },
      { H: 2, val: 0.5, train: 0, t: 4 },
    ];
    expect(meansByH(rs)).toEqual([
      { H: 2, mean: 0.5, count: 1 },
      { H: 8, mean: 0.2, count: 2 },
      { H: 64, mean: 0.2, count: 1 },
    ]);
    expect(betterThanFraction(rs, 0.15)).toBeCloseTo(0.75, 12);
    expect(betterThanFraction([], 0.1)).toBeNaN();
  });

  it('sanitizeRecords 는 깨진 항목을 버리고, toCsv 는 머리글과 행을 낸다', () => {
    const raw = [{ H: 8, val: 0.1, train: 0.01, t: 1700000000000 }, { H: 'x', val: 1 }, null, { H: 2, val: NaN, train: 0, t: 1 }];
    const rs = sanitizeRecords(raw);
    expect(rs).toHaveLength(1);
    expect(sanitizeRecords('nope')).toEqual([]);
    const csv = toCsv(rs);
    expect(csv.split('\n')[0]).toBe('time,hidden,train_mse,val_mse');
    expect(csv).toContain(',8,0.01,0.1');
  });
});
