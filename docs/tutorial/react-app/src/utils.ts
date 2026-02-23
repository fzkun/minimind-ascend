import { COLORS } from './constants';

export function softmax(arr: number[], temp?: number): number[] {
  const t = temp || 1;
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp((x - max) / t));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

export function mulberry32(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function tokenColor(i: number): string {
  return COLORS[i % COLORS.length];
}
