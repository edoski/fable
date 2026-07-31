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

export type ModelSelection = {
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
  execute(
    selection: ModelSelection,
    input: Float32Array,
  ): Promise<ModelOutput>;
  dispose(): Promise<void>;
};

type NativeModule = {
  load(source: ResourceSource): Promise<void>;
  forward(inputs: TensorPtr[]): Promise<TensorPtr[]>;
  delete(): void;
};

type NativeModuleFactory = () => NativeModule;

initExecutorch({ resourceFetcher: ExpoResourceFetcher });

export function createDefaultModelCatalog(): ModelCatalog {
  const manifest = require("../assets/models/manifest.json") as MobileManifest;
  const resources = {
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
  };
  return {
    chainManifest(chain) {
      return manifest.chains[chain];
    },
    select(chain, K) {
      return {
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
  let current: { artifactId: string; module: NativeModule } | null = null;
  let disposal: Promise<void> | null = null;
  const serialize = createSerialQueue();

  async function ensureLoaded(selection: ModelSelection): Promise<NativeModule> {
    const artifactId = selection.modelManifest.artifact_id;
    if (current?.artifactId === artifactId) return current.module;

    if (current !== null) {
      const previous = current.module;
      current = null;
      previous.delete();
    }

    const module = createNativeModule();
    try {
      await module.load(selection.source);
    } catch (error) {
      module.delete();
      throw error;
    }

    current = { artifactId, module };
    return module;
  }

  function execute(
    selection: ModelSelection,
    input: Float32Array,
  ): Promise<ModelOutput> {
    if (disposal !== null) throw new Error("Model runtime is disposed");
    return serialize(async () => {
      const module = await ensureLoaded(selection);
      const outputs = await module.forward([
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
      return decodeOutputs(outputs, selection.K);
    });
  }

  function dispose(): Promise<void> {
    if (disposal !== null) return disposal;
    disposal = serialize(async () => {
      if (current === null) return;
      const model = current;
      current = null;
      model.module.delete();
    });
    return disposal;
  }

  return { execute, dispose };
}

function decodeOutputs(
  outputs: readonly TensorPtr[],
  K: Horizon,
): ModelOutput {
  if (outputs.length !== 2) {
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
  tensor: TensorPtr,
  shape: readonly number[],
  label: string,
): Float32Array {
  if (tensor.scalarType !== ScalarType.FLOAT) {
    throw new Error(`${label} output must be float32`);
  }
  if (
    tensor.sizes.length !== shape.length ||
    tensor.sizes.some((size, index) => size !== shape[index])
  ) {
    throw new Error(
      `${label} output must have shape [${shape.join(", ")}]`,
    );
  }
  const values = new Float32Array(tensor.dataPtr as ArrayBuffer);
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
  return values.slice();
}
