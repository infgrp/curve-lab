/**
 * 전체 흐름과 키오스크 기능 (SPEC 4절, PROCEDURE Phase 7).
 *
 * 시작 → 1단계 → 2단계 → 3단계 → 결과. 각 단계에 건너뛰기.
 * 무입력 idleSeconds(기본 120초) 뒤 시작 화면으로 돌아가며 상태를 모두 버린다 (ui/idle.ts, 시간 주입).
 * 교사 설정: 화면 왼쪽 위 모서리를 3초 길게 누르면 열린다 — 시드 고정, σ, 학습 점 수, 기록 초기화, CSV 내보내기.
 */
import { type Dataset, TRUE_FN, generateDataset } from '../core/data.ts';
import { type HingeParams, hingePredict } from '../core/hinge.ts';
import { type LinearParams, linearLoss } from '../core/linear.ts';
import { hingeLoss } from '../core/hinge.ts';
import { seedFrom } from '../core/rng.ts';
import { toCsv } from '../core/scoreboard.ts';
import { DEFAULT_SETTINGS, SETTINGS_LIMITS, type Settings, sanitizeSettings } from '../core/settings.ts';
import { ensureBackground, legend } from './chrome.ts';
import { IdleTimer, type Scheduler, realScheduler, watchUserInput } from './idle.ts';
import { COLORS, Plot } from './plot.ts';
import { ScoreStore } from './scoreboard.ts';
import { type Stage1Handle, mountStage1 } from './stage1.ts';
import { type Stage2Handle, mountStage2 } from './stage2.ts';
import { type Stage3Handle, type Stage3Result, mountStage3 } from './stage3.ts';
import { T } from './text.ko.ts';

export type Screen = 'start' | 'stage1' | 'stage2' | 'stage3' | 'result';

export interface AppOptions {
  scheduler?: Scheduler;
  now?: () => number;
  /** 세션 시드의 엔트로피 (기본 crypto) */
  entropy?: () => number;
}

interface Session {
  data: Dataset;
  linear: LinearParams | null;
  hinge: HingeParams | null;
  net: Stage3Result | null;
  netPredict: ((x: number) => number) | null;
}

const SETTINGS_KEY = 'curve-lab:settings';
const HOLD_MS = 3000;

export class App {
  readonly root: HTMLElement;
  screen: Screen = 'start';
  settings: Settings;
  session: Session | null = null;
  readonly store: ScoreStore;
  private readonly now: () => number;
  private readonly entropy: () => number;
  private idle: IdleTimer;
  private unwatch: (() => void) | null = null;
  private handle: Stage1Handle | Stage2Handle | Stage3Handle | { destroy(): void } | null = null;
  private hotspotTimer: unknown = null;
  private readonly sched: Scheduler;
  private dialog: HTMLElement | null = null;

  constructor(root: HTMLElement, options: AppOptions = {}) {
    this.root = root;
    this.now = options.now ?? (() => Date.now());
    this.sched = options.scheduler ?? realScheduler;
    this.entropy = options.entropy ?? (() => {
      const a = new Uint32Array(1);
      crypto.getRandomValues(a);
      return a[0];
    });
    this.store = new ScoreStore(this.now);
    this.settings = this.loadSettings();
    this.idle = new IdleTimer(this.settings.idleSeconds * 1000, () => this.resetToStart(), this.sched);
    ensureBackground();
    this.mountHotspot();
    this.unwatch = watchUserInput(document, this.idle);
    this.showStart();
  }

  // ── 설정 저장소 (try/catch) ──
  private loadSettings(): Settings {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      return raw ? sanitizeSettings(JSON.parse(raw)) : { ...DEFAULT_SETTINGS };
    } catch {
      return { ...DEFAULT_SETTINGS };
    }
  }

  private saveSettings(s: Settings): void {
    this.settings = s;
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch {
      /* 저장 실패해도 이번 세션에는 적용된다 */
    }
    this.idle.stop();
    this.idle = new IdleTimer(s.idleSeconds * 1000, () => this.resetToStart(), this.sched);
    this.unwatch?.();
    this.unwatch = watchUserInput(document, this.idle);
  }

  // ── 화면 전환 ──
  private clear(): void {
    this.handle?.destroy();
    this.handle = null;
    this.root.innerHTML = '';
  }

  newSession(): Session {
    const seed = this.settings.fixedSeed ?? seedFrom(this.now(), this.entropy());
    const data = generateDataset(seed, { n: this.settings.n, sigma: this.settings.sigma });
    return { data, linear: null, hinge: null, net: null, netPredict: null };
  }

  resetToStart(): void {
    this.closeDialog();
    this.session = null;
    this.showStart();
  }

  showStart(): void {
    this.clear();
    this.screen = 'start';
    this.idle.stop(); // 시작 화면에서는 잴 필요가 없다
    const s = T.start;
    this.root.innerHTML = `
      <section class="start">
        <div class="glass start-card">
          <span class="step-chip">${s.kicker}</span>
          <h1 class="start-title">${s.title}</h1>
          <p class="lead">${s.lead}</p>
          <ol class="start-steps">${s.steps.map((x, i) => `<li><b>${i + 1}</b>${x}</li>`).join('')}</ol>
          <button class="btn primary start-btn" type="button" data-role="begin">${s.begin} →</button>
          <p class="hint">${s.time}</p>
        </div>
      </section>`;
    this.root.querySelector<HTMLButtonElement>('[data-role="begin"]')!.addEventListener('click', () => this.begin());
  }

  begin(): void {
    this.session = this.newSession();
    this.idle.start();
    this.showStage1();
  }

  showStage1(): void {
    const ss = this.session ?? (this.session = this.newSession());
    this.clear();
    this.screen = 'stage1';
    this.handle = mountStage1(this.root, ss.data, {
      onNext: (r) => {
        ss.linear = r;
        this.showStage2();
      },
    });
  }

  showStage2(): void {
    const ss = this.session ?? (this.session = this.newSession());
    this.clear();
    this.screen = 'stage2';
    const lin = ss.linear ?? { a: 0, b: 0 };
    this.handle = mountStage2(this.root, ss.data, lin, {
      onNext: (r) => {
        ss.hinge = r;
        this.showStage3();
      },
    });
  }

  showStage3(): void {
    const ss = this.session ?? (this.session = this.newSession());
    this.clear();
    this.screen = 'stage3';
    const h = mountStage3(this.root, ss.data, {
      store: this.store,
      onNext: (r) => {
        ss.net = r;
        ss.netPredict = r ? h.predictor : null;
        this.showResult();
      },
    });
    this.handle = h;
  }

  showResult(): void {
    const ss = this.session ?? (this.session = this.newSession());
    this.clear();
    this.screen = 'result';
    const r = T.result;
    const { data } = ss;
    const lin = ss.linear;
    const hinge = ss.hinge;
    const net = ss.net;
    const valMse = (f: (x: number) => number) => {
      let s = 0;
      for (let i = 0; i < data.val.x.length; i++) {
        const d = f(data.val.x[i]) - data.val.y[i];
        s += d * d;
      }
      return s / data.val.x.length;
    };
    const card = (title: string, trainV: number | null, valV: number | null, color: string) => `
      <div class="card stat result-card" style="border-left:8px solid ${color}">
        <span class="k">${title}</span>
        <div class="result-nums">
          <span><small>${r.trainMse}</small><b>${trainV === null ? r.skipped : trainV.toFixed(3)}</b></span>
          <span><small>${r.valMse}</small><b class="${valV !== null && valV < 2 * data.sigma * data.sigma ? 'ok' : ''}">${valV === null ? r.skipped : valV.toFixed(3)}</b></span>
        </div>
      </div>`;
    this.root.innerHTML = `
      <section class="stage result">
        <header class="stage-head"><div><div class="titlebar"><span class="step-chip">${r.title}</span></div><p class="lead">${r.lead}</p></div></header>
        <div class="stage-body">
          <div class="glass panel">
            <div class="panel-head"><h2>${T.app.legend.train} · 숨겨진 점 · 세 모델</h2>${legend([
              { kind: 'dot', color: COLORS.train, label: T.app.legend.train },
              { kind: 'diamond', color: COLORS.val, label: '숨겨진 점' },
              { kind: 'line', color: '#8794a6', label: r.s1 },
              { kind: 'line', color: '#7c3aed', label: r.s2 },
              { kind: 'line', color: COLORS.train, label: r.s3none },
              { kind: 'dash', color: '#15223a', label: r.trueCurve },
            ])}</div>
            <canvas class="plot" data-role="plot"></canvas>
            <div class="controls"></div>
          </div>
          <div class="glass panel">
            <div class="panel-head"><h2>손실 비교</h2><span class="hint">σ² = ${(data.sigma * data.sigma).toFixed(4)}</span></div>
            <div class="controls scroll result-list">
              ${card(r.s1, lin ? linearLoss(data.train, lin) : null, lin ? valMse((x) => lin.a * x + lin.b) : null, '#8794a6')}
              ${card(r.s2, hinge ? hingeLoss(data.train, hinge) : null, hinge ? valMse((x) => hingePredict(hinge, x)) : null, '#7c3aed')}
              ${card(net ? r.s3(net.H) : r.s3none, net ? net.trainMse : null, net ? net.valMse : null, COLORS.train)}
              <ul class="card takeaway">${r.takeaway.map((x) => `<li>${x}</li>`).join('')}</ul>
            </div>
          </div>
        </div>
        <footer class="stage-foot"><p class="bridge"></p><div class="row"><button class="btn primary" type="button" data-role="again">↻ ${r.again}</button></div></footer>
      </section>`;
    const plot = new Plot(this.root.querySelector<HTMLCanvasElement>('[data-role="plot"]')!, { xRange: [-1.2, 1.2], yRange: [-2.2, 2.2], xLabel: 'x', yLabel: 'y' });
    const draw = () => {
      plot.clear();
      plot.drawAxes();
      plot.drawCurve(TRUE_FN, { color: '#15223a', width: 2.5, dash: [8, 6] });
      if (lin) plot.drawCurve((x) => lin.a * x + lin.b, { color: '#8794a6', width: 3 });
      if (hinge) plot.drawCurve((x) => hingePredict(hinge, x), { color: '#7c3aed', width: 3.5 });
      if (ss.netPredict) plot.drawCurve(ss.netPredict, { color: COLORS.train, width: 4.5 });
      plot.drawPoints(data.val.x, data.val.y, { shape: 'diamond', color: COLORS.val, radius: 7 });
      plot.drawPoints(data.train.x, data.train.y, { shape: 'circle', color: COLORS.train, radius: 9 });
    };
    plot.onResize(draw);
    this.handle = { destroy: () => plot.destroy() };
    this.root.querySelector<HTMLButtonElement>('[data-role="again"]')!.addEventListener('click', () => this.resetToStart());
  }

  // ── 교사 설정: 왼쪽 위 모서리 3초 길게 누르기 ──
  private mountHotspot(): void {
    const hs = document.createElement('div');
    hs.className = 'hotspot';
    hs.setAttribute('aria-label', T.settings.title);
    const cancel = () => {
      if (this.hotspotTimer !== null) this.sched.cancel(this.hotspotTimer);
      this.hotspotTimer = null;
      hs.classList.remove('holding');
    };
    hs.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      cancel();
      hs.classList.add('holding');
      this.hotspotTimer = this.sched.schedule(() => {
        this.hotspotTimer = null;
        hs.classList.remove('holding');
        this.openSettings();
      }, HOLD_MS);
    });
    for (const e of ['pointerup', 'pointercancel', 'pointerleave'] as const) hs.addEventListener(e, cancel);
    document.body.append(hs);
  }

  openSettings(): void {
    this.closeDialog();
    const s = T.settings;
    const cur = this.settings;
    const today = this.store.today().length;
    const dlg = document.createElement('div');
    dlg.className = 'dialog-backdrop';
    dlg.innerHTML = `
      <div class="glass dialog" role="dialog" aria-modal="true" aria-label="${s.title}">
        <h2>${s.title}</h2>
        <p class="hint">${s.hint}</p>
        <label class="field"><span>${s.fixSeed}</span><input type="checkbox" data-role="fix" ${cur.fixedSeed !== null ? 'checked' : ''} /></label>
        <label class="field"><span>${s.seed}</span><input type="number" min="0" step="1" data-role="seed" value="${cur.fixedSeed ?? ''}" /></label>
        <p class="hint">${s.seedHint}</p>
        <label class="field"><span>${s.sigma}</span><input type="number" min="${SETTINGS_LIMITS.sigma[0]}" max="${SETTINGS_LIMITS.sigma[1]}" step="0.05" data-role="sigma" value="${cur.sigma}" /></label>
        <label class="field"><span>${s.n}</span><input type="number" min="${SETTINGS_LIMITS.n[0]}" max="${SETTINGS_LIMITS.n[1]}" step="1" data-role="n" value="${cur.n}" /></label>
        <label class="field"><span>${s.idle}</span><input type="number" min="${SETTINGS_LIMITS.idleSeconds[0]}" max="${SETTINGS_LIMITS.idleSeconds[1]}" step="10" data-role="idle" value="${cur.idleSeconds}" /></label>
        <div class="row" style="justify-content:space-between">
          <span class="badge" data-role="count">${s.records(today)}</span>
          <span class="row">
            <button class="btn ghost" type="button" data-role="clear">${s.clear}</button>
            <button class="btn ghost" type="button" data-role="export">${s.exportCsv}</button>
          </span>
        </div>
        <p class="status" data-role="msg"></p>
        <div class="row" style="justify-content:flex-end">
          <button class="btn" type="button" data-role="close">${s.close}</button>
          <button class="btn primary" type="button" data-role="save">${s.save}</button>
        </div>
      </div>`;
    const q = <E extends Element>(role: string) => dlg.querySelector<E>(`[data-role="${role}"]`)!;
    const msg = q<HTMLParagraphElement>('msg');
    q<HTMLButtonElement>('close').addEventListener('click', () => this.closeDialog());
    q<HTMLButtonElement>('clear').addEventListener('click', () => {
      this.store.clearToday();
      q<HTMLSpanElement>('count').textContent = s.records(0);
      msg.textContent = s.cleared;
    });
    q<HTMLButtonElement>('export').addEventListener('click', () => {
      const n = this.exportCsv();
      msg.textContent = n ? s.exported(n) : s.nothing;
    });
    q<HTMLButtonElement>('save').addEventListener('click', () => {
      const fix = q<HTMLInputElement>('fix').checked;
      const seedRaw = q<HTMLInputElement>('seed').value.trim();
      const next = sanitizeSettings({
        fixedSeed: fix && seedRaw !== '' ? seedRaw : null,
        sigma: q<HTMLInputElement>('sigma').value,
        n: q<HTMLInputElement>('n').value,
        idleSeconds: q<HTMLInputElement>('idle').value,
      });
      this.saveSettings(next);
      this.resetToStart();
    });
    document.body.append(dlg);
    this.dialog = dlg;
  }

  closeDialog(): void {
    this.dialog?.remove();
    this.dialog = null;
  }

  /** 저장된 모든 날짜의 기록을 CSV 파일로 내려받는다. 내보낸 건수를 돌려준다. */
  exportCsv(): number {
    const days = this.store.days();
    const all = days.flatMap((d) => this.store.forDay(d));
    const todayOnly = this.store.today();
    const records = all.length ? all : todayOnly;
    if (!records.length) return 0;
    const csv = toCsv(records);
    try {
      const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `curve-lab-records-${new Date(this.now()).toISOString().slice(0, 10)}.csv`;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch {
      /* 내려받기를 막는 환경이면 조용히 실패 */
    }
    return records.length;
  }

  destroy(): void {
    this.idle.stop();
    this.unwatch?.();
    this.closeDialog();
    this.clear();
  }
}
