import type { Hash } from "viem";

export function hashOf(value: bigint): Hash {
  return `0x${value.toString(16).padStart(64, "0")}`;
}

export async function flushMicrotasks(): Promise<void> {
  for (let index = 0; index < 10; index += 1) {
    await Promise.resolve();
  }
}
