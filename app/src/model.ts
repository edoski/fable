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

import {
  FEATURE_NAMES,
  type ChainManifest,
  type FeatureManifest,
  type FeatureName,
} from "./features";
import {
  SUPPORTED_CHAINS,
  type SupportedChain,
} from "./rpc";
import type { Horizon } from "./inference";

const MODEL_HORIZONS = [2, 3, 4, 5] as const satisfies readonly Horizon[];

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
  executorch_version: string;
  chains: Record<SupportedChain, MobileChainManifest>;
};

export type ModelResourceTable = Record<
  SupportedChain,
  Record<Horizon, number>
>;

export type ModelSelection = {
  chain: SupportedChain;
  K: Horizon;
  source: number;
  chainManifest: MobileChainManifest;
  modelManifest: ModelManifest;
};

export type ModelCatalog = {
  chainManifest(chain: SupportedChain): MobileChainManifest;
  select(chain: SupportedChain, K: Horizon): ModelSelection;
};

export type ModelOutput = {
  actionLogits: Float32Array;
  minimumFeeZ: number;
};

export class ModelOutputError extends Error {
  constructor(error: unknown) {
    super(
      error instanceof Error ? error.message : "Invalid model output",
      { cause: error },
    );
    this.name = "ModelOutputError";
  }
}

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

const EXECUTORCH_VERSION = "1.2.0";
const CHAIN_IDS: Record<SupportedChain, number> = {
  ethereum: 1,
  polygon: 137,
  avalanche: 43_114,
};
const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

const BUNDLED_MODEL_SOURCES = {
  ethereum: {
    2: () => require("../assets/models/ethereum-k2.pte"),
    3: () => require("../assets/models/ethereum-k3.pte"),
    4: () => require("../assets/models/ethereum-k4.pte"),
    5: () => require("../assets/models/ethereum-k5.pte"),
  },
  polygon: {
    2: () => require("../assets/models/polygon-k2.pte"),
    3: () => require("../assets/models/polygon-k3.pte"),
    4: () => require("../assets/models/polygon-k4.pte"),
    5: () => require("../assets/models/polygon-k5.pte"),
  },
  avalanche: {
    2: () => require("../assets/models/avalanche-k2.pte"),
    3: () => require("../assets/models/avalanche-k3.pte"),
    4: () => require("../assets/models/avalanche-k4.pte"),
    5: () => require("../assets/models/avalanche-k5.pte"),
  },
} as const;

initExecutorch({ resourceFetcher: ExpoResourceFetcher });

export function createDefaultModelCatalog(): ModelCatalog {
  return createModelCatalog(
    require("../assets/models/manifest.json"),
    {
      ethereum: {
        2: BUNDLED_MODEL_SOURCES.ethereum[2](),
        3: BUNDLED_MODEL_SOURCES.ethereum[3](),
        4: BUNDLED_MODEL_SOURCES.ethereum[4](),
        5: BUNDLED_MODEL_SOURCES.ethereum[5](),
      },
      polygon: {
        2: BUNDLED_MODEL_SOURCES.polygon[2](),
        3: BUNDLED_MODEL_SOURCES.polygon[3](),
        4: BUNDLED_MODEL_SOURCES.polygon[4](),
        5: BUNDLED_MODEL_SOURCES.polygon[5](),
      },
      avalanche: {
        2: BUNDLED_MODEL_SOURCES.avalanche[2](),
        3: BUNDLED_MODEL_SOURCES.avalanche[3](),
        4: BUNDLED_MODEL_SOURCES.avalanche[4](),
        5: BUNDLED_MODEL_SOURCES.avalanche[5](),
      },
    },
  );
}

export function createModelCatalog(
  manifestValue: unknown,
  resourceValue: unknown,
): ModelCatalog {
  const manifest = parseManifest(manifestValue);
  const resources = parseResources(resourceValue);

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
  let transitions: Promise<void> = Promise.resolve();

  function requireActive(): void {
    if (disposed) throw abortError("Model runtime is disposed");
  }

  function requireDesired(key: string): void {
    requireActive();
    if (desiredKey !== key) {
      throw abortError("Model selection changed");
    }
  }

  function exclusively<T>(operation: () => Promise<T>): Promise<T> {
    const result = transitions.then(operation, operation);
    transitions = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
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
    return exclusively(async () => {
      await ensureLoaded(selection, key);
    });
  }

  async function execute(
    selection: ModelSelection,
    input: Float32Array,
  ): Promise<ModelOutput> {
    validateInput(input, selection.chainManifest);
    requireActive();
    const key = selectionKey(selection);
    desiredKey = key;
    const outputs = await exclusively(async () => {
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
    try {
      return decodeOutputs(outputs, selection.K);
    } catch (error) {
      throw new ModelOutputError(error);
    }
  }

  async function dispose(): Promise<void> {
    if (disposed) {
      await transitions;
      return;
    }
    disposed = true;
    desiredKey = null;
    await exclusively(async () => {
      if (current === null) return;
      const model = current;
      current = null;
      model.module.delete();
    });
  }

  return { prepare, execute, dispose };
}

function parseManifest(value: unknown): MobileManifest {
  const manifest = exactObject(
    value,
    ["executorch_version", "chains"],
    "manifest",
  );
  if (manifest.executorch_version !== EXECUTORCH_VERSION) {
    throw new Error(
      `manifest executorch_version must be ${EXECUTORCH_VERSION}`,
    );
  }
  const chains = exactObject(
    manifest.chains,
    SUPPORTED_CHAINS,
    "manifest chains",
  );
  const parsed = {} as Record<SupportedChain, MobileChainManifest>;
  const artifactIds = new Set<string>();
  for (const chain of SUPPORTED_CHAINS) {
    parsed[chain] = parseChainManifest(
      chains[chain],
      chain,
      artifactIds,
    );
  }
  return {
    executorch_version: EXECUTORCH_VERSION,
    chains: parsed,
  };
}

function parseChainManifest(
  value: unknown,
  chain: SupportedChain,
  artifactIds: Set<string>,
): MobileChainManifest {
  const manifest = exactObject(
    value,
    ["chain_id", "context_blocks", "features", "models"],
    `${chain} manifest`,
  );
  if (manifest.chain_id !== CHAIN_IDS[chain]) {
    throw new Error(
      `${chain} chain_id must be ${CHAIN_IDS[chain]}`,
    );
  }
  const contextBlocks = positiveSafeInteger(
    manifest.context_blocks,
    `${chain} context_blocks`,
  );
  if (!Array.isArray(manifest.features) || manifest.features.length === 0) {
    throw new Error(`${chain} features must be a nonempty array`);
  }
  const features = manifest.features.map((feature, index) =>
    parseFeature(feature, chain, index),
  );
  const names = new Set(features.map((feature) => feature.name));
  if (names.size !== features.length) {
    throw new Error(`${chain} feature names must be unique`);
  }
  if (
    chain !== "ethereum" &&
    names.has("log_exact_forming_base_fee_per_gas")
  ) {
    throw new Error(
      "log_exact_forming_base_fee_per_gas is Ethereum-only",
    );
  }

  const modelValues = exactObject(
    manifest.models,
    MODEL_HORIZONS.map(String),
    `${chain} models`,
  );
  const models = {} as Record<Horizon, ModelManifest>;
  for (const K of MODEL_HORIZONS) {
    models[K] = parseModel(
      modelValues[String(K)],
      `${chain} K=${K}`,
      artifactIds,
    );
  }
  return {
    chain_id: CHAIN_IDS[chain],
    context_blocks: contextBlocks,
    features,
    models,
  };
}

function parseFeature(
  value: unknown,
  chain: SupportedChain,
  index: number,
): FeatureManifest {
  const feature = exactObject(
    value,
    ["name", "mean", "standard_deviation"],
    `${chain} feature ${index}`,
  );
  if (
    typeof feature.name !== "string" ||
    !FEATURE_NAMES.includes(feature.name as FeatureName)
  ) {
    throw new Error(`${chain} feature ${index} has an unsupported name`);
  }
  return {
    name: feature.name as FeatureName,
    mean: finiteNumber(feature.mean, `${chain} feature ${index} mean`),
    standard_deviation: positiveFiniteNumber(
      feature.standard_deviation,
      `${chain} feature ${index} standard_deviation`,
    ),
  };
}

function parseModel(
  value: unknown,
  label: string,
  artifactIds: Set<string>,
): ModelManifest {
  const model = exactObject(value, ["artifact_id", "target"], label);
  if (
    typeof model.artifact_id !== "string" ||
    !UUID_V4.test(model.artifact_id)
  ) {
    throw new Error(`${label} artifact_id must be a UUIDv4`);
  }
  if (artifactIds.has(model.artifact_id)) {
    throw new Error("manifest artifact IDs must be unique");
  }
  artifactIds.add(model.artifact_id);
  const target = exactObject(
    model.target,
    ["mean", "standard_deviation"],
    `${label} target`,
  );
  return {
    artifact_id: model.artifact_id,
    target: {
      mean: finiteNumber(target.mean, `${label} target mean`),
      standard_deviation: positiveFiniteNumber(
        target.standard_deviation,
        `${label} target standard_deviation`,
      ),
    },
  };
}

function parseResources(value: unknown): ModelResourceTable {
  const resources = exactObject(
    value,
    SUPPORTED_CHAINS,
    "model resources",
  );
  const parsed = {} as ModelResourceTable;
  for (const chain of SUPPORTED_CHAINS) {
    const cells = exactObject(
      resources[chain],
      MODEL_HORIZONS.map(String),
      `${chain} model resources`,
    );
    parsed[chain] = {} as Record<Horizon, number>;
    for (const K of MODEL_HORIZONS) {
      const source = cells[String(K)];
      if (
        typeof source !== "number" ||
        !Number.isSafeInteger(source) ||
        source < 0
      ) {
        throw new Error(
          `${chain} K=${K} resource must be a static Metro asset`,
        );
      }
      parsed[chain][K] = source;
    }
  }
  return parsed;
}

function exactObject(
  value: unknown,
  keys: readonly string[],
  label: string,
): Record<string, unknown> {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  ) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(`${label} must contain exactly ${expected.join(", ")}`);
  }
  return value as Record<string, unknown>;
}

function positiveSafeInteger(value: unknown, label: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value <= 0
  ) {
    throw new Error(`${label} must be a positive safe integer`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
  return value;
}

function positiveFiniteNumber(value: unknown, label: string): number {
  const number = finiteNumber(value, label);
  if (number <= 0) throw new Error(`${label} must be positive`);
  return number;
}

function selectionKey(selection: ModelSelection): string {
  return `${selection.chain}:${selection.K}:${selection.modelManifest.artifact_id}`;
}

function validateInput(
  input: Float32Array,
  manifest: MobileChainManifest,
): void {
  const expected =
    manifest.context_blocks * manifest.features.length;
  if (!(input instanceof Float32Array) || input.length !== expected) {
    throw new Error(
      `Model input must contain exactly ${expected} float32 values`,
    );
  }
  for (const value of input) {
    if (!Number.isFinite(value)) {
      throw new Error("Model input values must be finite");
    }
  }
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
