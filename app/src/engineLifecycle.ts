import type {
  ChainSnapshot,
  InferenceEngine,
} from "./inference";

export type RpcStatus = "checking" | "live" | "offline";

export type EngineLifecycleObserver = {
  onConstructionError(error: unknown): void;
  onDisposalError(error: unknown): void;
  onRpcUnavailable(error: unknown): void;
  onSnapshot(engine: InferenceEngine, snapshot: ChainSnapshot): void;
  onStatus(status: RpcStatus): void;
};

type EngineRecord = {
  engine: InferenceEngine;
  revision: number;
  stopPolling: () => void;
};

export type EngineLifecycle = {
  replace(create: () => InferenceEngine): Promise<InferenceEngine | null>;
  release(lease: Promise<InferenceEngine | null>): Promise<void>;
};

export function createEngineLifecycle(
  observer: EngineLifecycleObserver,
): EngineLifecycle {
  let current: EngineRecord | null = null;
  let replacementRevision = 0;
  let transitions: Promise<void> = Promise.resolve();

  function serialize<Result>(
    operation: () => Promise<Result>,
  ): Promise<Result> {
    const result = transitions.then(operation, operation);
    transitions = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  async function dispose(record: EngineRecord): Promise<void> {
    try {
      record.stopPolling();
    } catch (error) {
      observer.onDisposalError(error);
    }
    try {
      await record.engine.dispose();
    } catch (error) {
      observer.onDisposalError(error);
    }
  }

  function replace(
    create: () => InferenceEngine,
  ): Promise<InferenceEngine | null> {
    replacementRevision += 1;
    const revision = replacementRevision;
    observer.onStatus("checking");
    return serialize(async () => {
      if (current !== null) {
        const previous = current;
        current = null;
        await dispose(previous);
      }
      if (revision !== replacementRevision) return null;

      let engine: InferenceEngine;
      try {
        engine = create();
      } catch (error) {
        observer.onConstructionError(error);
        return null;
      }

      const record: EngineRecord = {
        engine,
        revision,
        stopPolling: () => undefined,
      };
      current = record;
      try {
        record.stopPolling = engine.startPolling(
          (snapshot) => {
            if (
              current !== record ||
              record.revision !== replacementRevision
            ) {
              return;
            }
            observer.onStatus("live");
            observer.onSnapshot(engine, snapshot);
          },
          (error) => {
            if (
              current !== record ||
              record.revision !== replacementRevision
            ) {
              return;
            }
            observer.onStatus("offline");
            observer.onRpcUnavailable(error);
          },
        );
      } catch (error) {
        current = null;
        await dispose(record);
        observer.onConstructionError(error);
        return null;
      }
      return engine;
    });
  }

  function release(
    lease: Promise<InferenceEngine | null>,
  ): Promise<void> {
    return lease.then((engine) => {
      if (engine === null) return;
      return serialize(async () => {
        if (current?.engine !== engine) return;
        const released = current;
        current = null;
        await dispose(released);
      });
    });
  }

  return { replace, release };
}
