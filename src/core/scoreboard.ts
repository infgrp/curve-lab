/**
 * 방문자 분포 계산 (SPEC 3단계 "방문자 분포"). 저장(localStorage)은 ui/scoreboard.ts 가 맡고,
 * 여기는 날짜 키·히스토그램·뉴런 수별 평균 같은 순수 계산만 둔다.
 */

export interface ScoreRecord {
  /** 은닉 뉴런 수 */
  H: number;
  /** 검증 MSE */
  val: number;
  /** 학습 MSE */
  train: number;
  /** 기록 시각 (ms) */
  t: number;
}

/** 로컬 날짜 기준 'YYYY-MM-DD' */
export function dayKey(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

/** [lo, hi] 를 bins 칸으로 나눈 도수. hi 이상은 마지막 칸에 넣는다. */
export function binCounts(values: ArrayLike<number>, lo: number, hi: number, bins: number): number[] {
  const counts = new Array<number>(bins).fill(0);
  const w = (hi - lo) / bins;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v) || v < lo) continue;
    const k = Math.min(bins - 1, Math.floor((v - lo) / w));
    counts[k]++;
  }
  return counts;
}

/** 값이 들어가는 칸 번호 (binCounts 와 같은 규칙). 범위 아래면 -1 */
export function binIndex(v: number, lo: number, hi: number, bins: number): number {
  if (!Number.isFinite(v) || v < lo) return -1;
  return Math.min(bins - 1, Math.floor((v - lo) / ((hi - lo) / bins)));
}

/** 뉴런 수별 검증 MSE 평균과 개수 (H 오름차순) */
export function meansByH(records: readonly ScoreRecord[]): { H: number; mean: number; count: number }[] {
  const acc = new Map<number, { sum: number; n: number }>();
  for (const r of records) {
    if (!Number.isFinite(r.val)) continue;
    const a = acc.get(r.H) ?? { sum: 0, n: 0 };
    a.sum += r.val;
    a.n++;
    acc.set(r.H, a);
  }
  return [...acc.entries()].sort((p, q) => p[0] - q[0]).map(([H, a]) => ({ H, mean: a.sum / a.n, count: a.n }));
}

/** 내 값보다 작은(더 좋은) 기록의 비율 → 백분위 안내용. 기록이 없으면 NaN */
export function betterThanFraction(records: readonly ScoreRecord[], val: number): number {
  const vals = records.map((r) => r.val).filter(Number.isFinite);
  if (!vals.length) return NaN;
  let worse = 0;
  for (const v of vals) if (v > val) worse++;
  return worse / vals.length;
}

/** CSV (교사 설정의 내보내기용). 개인정보 없음. */
export function toCsv(records: readonly ScoreRecord[]): string {
  const lines = ['time,hidden,train_mse,val_mse'];
  for (const r of records) lines.push(`${new Date(r.t).toISOString()},${r.H},${r.train},${r.val}`);
  return lines.join('\n') + '\n';
}

/** 저장된 JSON 을 검사해 올바른 기록만 남긴다 */
export function sanitizeRecords(raw: unknown): ScoreRecord[] {
  if (!Array.isArray(raw)) return [];
  const out: ScoreRecord[] = [];
  for (const r of raw) {
    if (!r || typeof r !== 'object') continue;
    const o = r as Record<string, unknown>;
    const H = Number(o['H']);
    const val = Number(o['val']);
    const train = Number(o['train']);
    const t = Number(o['t']);
    if (Number.isInteger(H) && H > 0 && Number.isFinite(val) && Number.isFinite(train) && Number.isFinite(t)) out.push({ H, val, train, t });
  }
  return out;
}
