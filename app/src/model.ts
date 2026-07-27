import {
  ExecutorchModule,
  ScalarType,
  initExecutorch,
} from "react-native-executorch";
import type {
  ResourceSource,
  TensorPtr,
} from "react-native-executorch";
import { ExpoResourceFetcher } from "react-native-executorch-expo-resource-fetcher";

import type { ChainManifest } from "./features";
import type { Chain, Horizon } from "./domain";
import { createSerialQueue } from "./serialQueue";

export type TargetManifest = {
  mean: number;
  standard_deviation: number;
};

export type ModelManifest = {
  artifact_id: string;
  target: TargetManifest;
};

export type MobileChainManifest = ChainManifest & {
  models: Record<Horizon, ModelManifest>;
};

export type MobileManifest = {
  chains: Record<Chain, MobileChainManifest>;
};

export type ModelResourceTable = Record<
  Chain,
  Record<Horizon, number>
>;

export type ModelSelection = {
  chain: Chain;
  K: Horizon;
  source: number;
  chainManifest: MobileChainManifest;
  modelManifest: ModelManifest;
};

export type ModelCatalog = {
  chainManifest(chain: Chain): MobileChainManifest;
  select(chain: Chain, K: Horizon): ModelSelection;
};

export type ModelOutput = {
  actionLogits: Float32Array;
  minimumFeeZ: number;
};

export type ModelRuntime = {
  prepare(selection: ModelSelection): Promise<void>;
  execute(
    selection: ModelSelection,
    input: Float32Array,
  ): Promise<ModelOutput>;
  dispose(): Promise<void>;
};

type NativeModule = {
  load(source: ResourceSource): Promise<void>;
  forward(inputs: TensorPtr[]): Promise<unknown>;
  delete(): void;
};

type NativeModuleFactory = () => NativeModule;

type LoadedModel = {
  key: string;
  module: NativeModule;
};

initExecutorch({ resourceFetcher: ExpoResourceFetcher });

export function createDefaultModelCatalog(): ModelCatalog {
  return createModelCatalog(
    require("../assets/models/manifest.json") as MobileManifest,
    {
      ethereum: {
        2: require("../assets/models/ethereum-k2.pte"),
        3: require("../assets/models/ethereum-k3.pte"),
        4: require("../assets/models/ethereum-k4.pte"),
        5: require("../assets/models/ethereum-k5.pte"),
      },
      polygon: {
        2: require("../assets/models/polygon-k2.pte"),
        3: require("../assets/models/polygon-k3.pte"),
        4: require("../assets/models/polygon-k4.pte"),
        5: require("../assets/models/polygon-k5.pte"),
      },
      avalanche: {
        2: require("../assets/models/avalanche-k2.pte"),
        3: require("../assets/models/avalanche-k3.pte"),
        4: require("../assets/models/avalanche-k4.pte"),
        5: require("../assets/models/avalanche-k5.pte"),
      },
    },
  );
}

export function createModelCatalog(
  manifest: MobileManifest,
  resources: ModelResourceTable,
): ModelCatalog {
  return {
    chainManifest(chain) {
      return manifest.chains[chain];
    },
    select(chain, K) {
      return {
        chain,
        K,
        source: resources[chain][K],
        chainManifest: manifest.chains[chain],
        modelManifest: manifest.chains[chain].models[K],
      };
    },
  };
}

export function createModelRuntime(
  createNativeModule: NativeModuleFactory = () => new ExecutorchModule(),
): ModelRuntime {
  let current: LoadedModel | null = null;
  let desiredKey: string | null = null;
  let disposed = false;
  let disposal: Promise<void> | null = null;
  const serialize = createSerialQueue();

  function requireActive(): void {
    if (disposed) throw abortError("Model runtime is disposed");
  }

  function requireDesired(key: string): void {
    requireActive();
    if (desiredKey !== key) {
      throw abortError("Model selection changed");
    }
  }

  async function ensureLoaded(
    selection: ModelSelection,
    key: string,
  ): Promise<LoadedModel> {
    requireDesired(key);
    if (current?.key === key) return current;

    if (current !== null) {
      const previous = current;
      current = null;
      previous.module.delete();
    }

    requireDesired(key);
    const module = createNativeModule();
    try {
      await module.load(selection.source);
      requireDesired(key);
    } catch (error) {
      module.delete();
      throw error;
    }

    const loaded: LoadedModel = {
      key,
      module,
    };
    current = loaded;
    return loaded;
  }

  function prepare(selection: ModelSelection): Promise<void> {
    requireActive();
    const key = selectionKey(selection);
    desiredKey = key;
    return serialize(async () => {
      await ensureLoaded(selection, key);
    });
  }

  async function execute(
    selection: ModelSelection,
    input: Float32Array,
  ): Promise<ModelOutput> {
    requireActive();
    const key = selectionKey(selection);
    desiredKey = key;
    const outputs = await serialize(async () => {
      const model = await ensureLoaded(selection, key);
      const result = await model.module.forward([
        {
          dataPtr: input,
          sizes: [
            1,
            selection.chainManifest.context_blocks,
            selection.chainManifest.features.length,
          ],
          scalarType: ScalarType.FLOAT,
        },
      ]);
      requireDesired(key);
      return result;
    });
    return decodeOutputs(outputs, selection.K);
  }

  function dispose(): Promise<void> {
    if (disposal !== null) return disposal;
    disposed = true;
    desiredKey = null;
    disposal = serialize(async () => {
      if (current === null) return;
      const model = current;
      current = null;
      model.module.delete();
    });
    return disposal;
  }

  return { prepare, execute, dispose };
}

function selectionKey(selection: ModelSelection): string {
  return `${selection.chain}:${selection.K}:${selection.modelManifest.artifact_id}`;
}

function decodeOutputs(outputs: unknown, K: Horizon): ModelOutput {
  if (!Array.isArray(outputs) || outputs.length !== 2) {
    throw new Error(
      "ExecuTorch model must return exactly two float32 tensors",
    );
  }
  const actionLogits = readFloatTensor(
    outputs[0],
    [1, K],
    "action logits",
  );
  const minimumFee = readFloatTensor(
    outputs[1],
    [1],
    "minimum fee z",
  );
  return {
    actionLogits,
    minimumFeeZ: minimumFee[0],
  };
}

function readFloatTensor(
  value: unknown,
  shape: readonly number[],
  label: string,
): Float32Array {
  if (typeof value !== "object" || value === null) {
    throw new Error(`${label} output must be a tensor`);
  }
  const tensor = value as Partial<TensorPtr>;
  if (tensor.scalarType !== ScalarType.FLOAT) {
    throw new Error(`${label} output must be float32`);
  }
  if (
    !Array.isArray(tensor.sizes) ||
    tensor.sizes.length !== shape.length ||
    tensor.sizes.some((size, index) => size !== shape[index])
  ) {
    throw new Error(
      `${label} output must have shape [${shape.join(", ")}]`,
    );
  }

  let values: Float32Array;
  if (tensor.dataPtr instanceof Float32Array) {
    values = tensor.dataPtr;
  } else if (tensor.dataPtr instanceof ArrayBuffer) {
    if (tensor.dataPtr.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`${label} output has an invalid float32 buffer`);
    }
    values = new Float32Array(tensor.dataPtr);
  } else {
    throw new Error(`${label} output must contain float32 data`);
  }
  const expectedLength = shape.reduce(
    (size, dimension) => size * dimension,
    1,
  );
  if (values.length !== expectedLength) {
    throw new Error(
      `${label} output must contain exactly ${expectedLength} values`,
    );
  }
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error(`${label} output values must be finite`);
    }
  }
  return new Float32Array(values);
}

function abortError(message: string): Error {
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}
