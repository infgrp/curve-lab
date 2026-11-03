/// <reference types="vitest/config" />
import { defineConfig } from 'vite';

// 키오스크는 오프라인에서 돈다: 외부 CDN·원격 폰트·분석 스크립트 없음 (CLAUDE.md 기술 스택).
export default defineConfig({
  base: './',
  build: { target: 'es2022', sourcemap: false },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node', // src/core 는 DOM 에 의존하지 않는다
  },
});
