import { describe, expect, it, vi } from "vitest";

import { createEngineLifecycle } from "../src/engineLifecycle";
import type {
  ChainSnapshot,
  InferenceEngine,
} from "../src/inference";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

function snapshot(headBlock: number): ChainSnapshot {
  return {
    chain: "ethereum",
    head_block: headBlock,
    current_base_fee_per_gas: 20,
  };
}

function engine(dispose: () => Promise<void> = async () => undefined) {
  const stop = vi.fn();
  const value: InferenceEngine = {
    prepare: vi.fn(async () => undefined),
    startPolling: vi.fn(() => stop),
    run: vi.fn(async () => {
      throw new Error("unused");
    }),
    resolveOutcome: vi.fn(async () => {
      throw new Error("unused");
    }),
    dispose: vi.fn(dispose),
  };
  return {
    value,
    stop,
  };
}

function observer() {
  const constructionErrors: unknown[] = [];
  const disposalErrors: unknown[] = [];
  const rpcErrors: unknown[] = [];
  const snapshots: ChainSnapshot[] = [];
  const statuses: string[] = [];
  return {
    constructionErrors,
    disposalErrors,
    rpcErrors,
    snapshots,
    statuses,
    onConstructionError(error: unknown) {
      constructionErrors.push(error);
    },
    onDisposalError(error: unknown) {
      disposalErrors.push(error);
    },
    onRpcUnavailable(error: unknown) {
      rpcErrors.push(error);
    },
    onSnapshot(_engine: InferenceEngine, snapshot: ChainSnapshot) {
      snapshots.push(snapshot);
    },
    onStatus(status: string) {
      statuses.push(status);
    },
  };
}

describe("engine lifecycle", () => {
  it("waits for the old engine to dispose before constructing its replacement", async () => {
    const disposing = deferred<void>();
    const first = engine(() => disposing.promise);
    const second = engine();
    const events = observer();
    const lifecycle = createEngineLifecycle(events);
    const firstLease = lifecycle.replace(() => first.value);
    await expect(firstLease).resolves.toBe(first.value);
    const createSecond = vi.fn(() => second.value);

    const secondLease = lifecycle.replace(createSecond);
    await vi.waitFor(() => expect(first.value.dispose).toHaveBeenCalledOnce());
    expect(createSecond).not.toHaveBeenCalled();

    disposing.resolve();
    await expect(secondLease).resolves.toBe(second.value);
    expect(createSecond).toHaveBeenCalledOnce();
    await lifecycle.release(secondLease);
  });

  it("handles rejecting disposal during replacement and cleanup", async () => {
    const replacementError = new Error("replacement dispose failed");
    const cleanupError = new Error("cleanup dispose failed");
    const first = engine(async () => {
      throw replacementError;
    });
    const second = engine(async () => {
      throw cleanupError;
    });
    const events = observer();
    const lifecycle = createEngineLifecycle(events);
    await lifecycle.replace(() => first.value);

    const secondLease = lifecycle.replace(() => second.value);
    await expect(secondLease).resolves.toBe(second.value);
    await expect(lifecycle.release(secondLease)).resolves.toBeUndefined();

    expect(events.disposalErrors).toEqual([
      replacementError,
      cleanupError,
    ]);
    expect(first.stop).toHaveBeenCalledOnce();
    expect(second.stop).toHaveBeenCalledOnce();
  });

  it("keeps RPC checking on construction failure and changes it only from polling", async () => {
    const constructionError = new Error("catalog invalid");
    const events = observer();
    const lifecycle = createEngineLifecycle(events);

    await expect(
      lifecycle.replace(() => {
        throw constructionError;
      }),
    ).resolves.toBeNull();
    expect(events.statuses).toEqual(["checking"]);
    expect(events.constructionErrors).toEqual([constructionError]);

    const next = engine();
    let publishSnapshot: ((snapshot: ChainSnapshot) => void) | undefined;
    let publishError: ((error: unknown) => void) | undefined;
    next.value.startPolling = vi.fn((onSnapshot, onError) => {
      publishSnapshot = onSnapshot;
      publishError = onError;
      return next.stop;
    });
    const lease = lifecycle.replace(() => next.value);
    await lease;
    const rpcError = new Error("RPC unavailable");
    publishError?.(rpcError);
    publishSnapshot?.(snapshot(12));

    expect(events.statuses).toEqual([
      "checking",
      "checking",
      "offline",
      "live",
    ]);
    expect(events.rpcErrors).toEqual([rpcError]);
    expect(events.snapshots).toEqual([snapshot(12)]);
    await lifecycle.release(lease);
  });

  it("keeps the latest replacement active across stale callbacks and old cleanup", async () => {
    const stopError = new Error("stop polling failed");
    const first = engine();
    const skipped = engine();
    const replacement = engine();
    const events = observer();
    const lifecycle = createEngineLifecycle(events);
    let publishFirst: ((snapshot: ChainSnapshot) => void) | undefined;
    let failFirst: ((error: unknown) => void) | undefined;
    let publishReplacement: ((snapshot: ChainSnapshot) => void) | undefined;
    first.value.startPolling = vi.fn((onSnapshot, onError) => {
      publishFirst = onSnapshot;
      failFirst = onError;
      return () => {
        throw stopError;
      };
    });
    replacement.value.startPolling = vi.fn((onSnapshot) => {
      publishReplacement = onSnapshot;
      return replacement.stop;
    });
    const firstLease = lifecycle.replace(() => first.value);
    await firstLease;
    const createSkipped = vi.fn(() => skipped.value);

    const skippedLease = lifecycle.replace(createSkipped);
    const replacementLease = lifecycle.replace(() => replacement.value);
    const releaseFirst = lifecycle.release(firstLease);
    await expect(skippedLease).resolves.toBeNull();
    await expect(replacementLease).resolves.toBe(replacement.value);
    await expect(releaseFirst).resolves.toBeUndefined();

    expect(createSkipped).not.toHaveBeenCalled();
    expect(events.disposalErrors).toEqual([stopError]);
    expect(replacement.value.dispose).not.toHaveBeenCalled();
    const statuses = [...events.statuses];
    const snapshots = [...events.snapshots];
    failFirst?.(new Error("stale RPC failure"));
    publishFirst?.(snapshot(11));
    expect(events.statuses).toEqual(statuses);
    expect(events.snapshots).toEqual(snapshots);

    publishReplacement?.(snapshot(12));
    expect(events.statuses).toEqual([...statuses, "live"]);
    expect(events.snapshots).toEqual([snapshot(12)]);
    await lifecycle.release(replacementLease);
  });
});
