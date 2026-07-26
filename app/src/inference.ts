import { buildModelInput } from "./features";
import {
  createDefaultModelCatalog,
  createModelRuntime,
  ModelOutputError,
} from "./model";
import type {
  ModelCatalog,
  ModelOutput,
  ModelRuntime,
  ModelSelection,
} from "./model";
import {
  createChainSession,
  defaultRpcUrl,
} from "./rpc";
import type { ChainSession } from "./rpc";

export const CHAINS = ["ethereum", "polygon", "avalanche"] as const;
export type Chain = (typeof CHAINS)[number];

export const HORIZONS = [2, 3, 4, 5] as const;
export type Horizon = (typeof HORIZONS)[number];

export const CHAIN_DETAILS: Record<Chain, { label: string }> = {
  ethereum: { label: "Ethereum" },
  polygon: { label: "Polygon" },
  avalanche: { label: "Avalanche" },
};

export type InferenceResult = {
  chain: Chain;
  K: Horizon;
  artifact_id: string;
  head_block: number;
  head_hash: string;
  head_base_fee_per_gas: number;
  selected_action_k: number;
  immediate_block: number;
  target_block: number;
  predicted_minimum_base_fee_per_gas: number;
};

export type ChainSnapshot = {
  chain: Chain;
  head_block: number;
  current_base_fee_per_gas: number;
};

export type InferenceOutcome = {
  chain: Chain;
  immediate_block: number;
  selected_block: number;
  immediate_base_fee_per_gas: number;
  selected_base_fee_per_gas: number;
};

export type InferenceEngine = {
  prepare(K: Horizon): Promise<void>;
  snapshot(): Promise<ChainSnapshot>;
  startPolling(
    onSnapshot: (snapshot: ChainSnapshot) => void,
    onError?: (error: unknown) => void,
  ): () => void;
  run(K: Horizon): Promise<InferenceResult>;
  resolveOutcome(
    immediateBlock: number,
    selectedBlock: number,
  ): Promise<InferenceOutcome>;
  dispose(): Promise<void>;
};

export type InferenceEngineDependencies = {
  chain: Chain;
  catalog: ModelCatalog;
  model: ModelRuntime;
  session: ChainSession;
};

export function createInferenceEngine(chain: Chain): InferenceEngine;
export function createInferenceEngine(
  dependencies: InferenceEngineDependencies,
): InferenceEngine;
export function createInferenceEngine(
  input: Chain | InferenceEngineDependencies,
): InferenceEngine {
  const { chain, catalog, model, session } =
    typeof input === "string" ? defaultDependencies(input) : input;
  let selectedHorizon: Horizon | null = null;
  let selectionRevision = 0;
  let disposed = false;
  let disposal: Promise<void> | null = null;

  function beginSelection(K: Horizon): number {
    requireActive();
    if (selectedHorizon !== K) {
      selectedHorizon = K;
      selectionRevision += 1;
    }
    return selectionRevision;
  }

  function requireActive(): void {
    if (disposed) throw abortError("Inference engine is disposed");
  }

  function requireCurrent(revision: number): void {
    requireActive();
    if (revision !== selectionRevision) {
      throw abortError("Inference selection changed");
    }
  }

  async function prepare(K: Horizon): Promise<void> {
    const revision = beginSelection(K);
    const selection = catalog.select(chain, K);
    await prepareSelection(selection);
    requireCurrent(revision);
  }

  async function snapshot(): Promise<ChainSnapshot> {
    requireActive();
    const head = await session.readSnapshot();
    requireActive();
    return {
      chain,
      head_block: safeBigInt(head.number, "head block"),
      current_base_fee_per_gas: safeBigInt(
        head.baseFeePerGas,
        "current base fee",
      ),
    };
  }

  async function run(K: Horizon): Promise<InferenceResult> {
    const revision = beginSelection(K);
    const selection = catalog.select(chain, K);
    try {
      await model.prepare(selection);
    } catch (error) {
      throw inferenceFailure(
        "Could not load the selected model.",
        error,
      );
    }
    requireCurrent(revision);
    let context;
    try {
      context = await session.sync();
    } catch (error) {
      throw inferenceFailure(
        "Could not read the selected chain.",
        error,
      );
    }
    requireCurrent(revision);

    const head = context.blocks[context.blocks.length - 1];
    if (head === undefined || head.number !== context.head) {
      throw inferenceFailure(
        "Chain data is incomplete or invalid.",
        new Error("Synchronized context must end at the exact head"),
      );
    }
    let input: Float32Array;
    try {
      input = buildModelInput(
        context.blocks,
        context.feeHistory,
        selection.chainManifest,
      );
    } catch (error) {
      throw inferenceFailure(
        "Chain data is incomplete or invalid.",
        error,
      );
    }
    let output: ModelOutput;
    try {
      output = await model.execute(selection, input);
    } catch (error) {
      throw inferenceFailure(
        error instanceof ModelOutputError
          ? "The selected model returned invalid output."
          : "Could not run the selected model.",
        error,
      );
    }
    requireCurrent(revision);
    let prediction: ReturnType<typeof decodePrediction>;
    try {
      prediction = decodePrediction(selection, output);
    } catch (error) {
      throw inferenceFailure(
        "The selected model returned invalid output.",
        error,
      );
    }
    try {
      return createInferenceResult(chain, selection, head, prediction);
    } catch (error) {
      throw inferenceFailure(
        "Chain data is incomplete or invalid.",
        error,
      );
    }
  }

  function startPolling(
    onSnapshot: (snapshot: ChainSnapshot) => void,
    onError?: (error: unknown) => void,
  ): () => void {
    requireActive();
    return session.startPolling((block) => {
      requireActive();
      onSnapshot({
        chain,
        head_block: safeBigInt(block.number, "head block"),
        current_base_fee_per_gas: safeBigInt(
          block.baseFeePerGas,
          "current base fee",
        ),
      });
    }, onError);
  }

  async function resolveOutcome(
    immediateBlock: number,
    selectedBlock: number,
  ): Promise<InferenceOutcome> {
    requireActive();
    const immediate = safeBlockInput(immediateBlock, "immediate block");
    const selected = safeBlockInput(selectedBlock, "selected block");
    const outcome = await session.readOutcome(immediate, selected);
    requireActive();
    if (
      outcome.immediateBlock !== immediate ||
      outcome.selectedBlock !== selected
    ) {
      throw new Error("Chain outcome does not match the requested blocks");
    }
    return {
      chain,
      immediate_block: safeBigInt(
        outcome.immediateBlock,
        "immediate block",
      ),
      selected_block: safeBigInt(
        outcome.selectedBlock,
        "selected block",
      ),
      immediate_base_fee_per_gas: safeBigInt(
        outcome.immediateBaseFeePerGas,
        "immediate base fee",
      ),
      selected_base_fee_per_gas: safeBigInt(
        outcome.selectedBaseFeePerGas,
        "selected base fee",
      ),
    };
  }

  function dispose(): Promise<void> {
    if (disposal !== null) return disposal;
    disposed = true;
    selectionRevision += 1;
    let sessionError: unknown;
    try {
      session.dispose();
    } catch (error) {
      sessionError = error;
    }
    disposal = model.dispose().then(() => {
      if (sessionError !== undefined) throw sessionError;
    });
    return disposal;
  }

  async function prepareSelection(
    selection: ModelSelection,
  ): Promise<void> {
    const [chainResult, modelResult] = await Promise.allSettled([
      session.sync(),
      model.prepare(selection),
    ]);
    if (chainResult.status === "rejected") throw chainResult.reason;
    if (modelResult.status === "rejected") throw modelResult.reason;
  }

  return {
    prepare,
    snapshot,
    startPolling,
    run,
    resolveOutcome,
    dispose,
  };
}

function defaultDependencies(chain: Chain): InferenceEngineDependencies {
  const catalog = createDefaultModelCatalog();
  const manifest = catalog.chainManifest(chain);
  return {
    chain,
    catalog,
    model: createModelRuntime(),
    session: createChainSession({
      chain,
      rpcUrl: defaultRpcUrl(chain),
      contextBlocks: manifest.context_blocks,
      orderedFeatures: manifest.features.map((feature) => feature.name),
    }),
  };
}

function decodePrediction(
  selection: ModelSelection,
  output: ModelOutput,
): {
  selectedAction: number;
  predictedFee: number;
} {
  if (output.actionLogits.length !== selection.K) {
    throw new Error(
      `Model action logits must contain exactly ${selection.K} values`,
    );
  }
  let action = 0;
  for (let index = 0; index < output.actionLogits.length; index += 1) {
    const value = output.actionLogits[index];
    if (!Number.isFinite(value)) {
      throw new Error("Model action logits must be finite");
    }
    if (value > output.actionLogits[action]) action = index;
  }
  if (
    !Number.isSafeInteger(action) ||
    action < 0 ||
    action >= selection.K
  ) {
    throw new Error(`Model action must be between 0 and ${selection.K - 1}`);
  }
  if (!Number.isFinite(output.minimumFeeZ)) {
    throw new Error("Model minimum fee z must be finite");
  }

  const target = selection.modelManifest.target;
  const predictedFee = Math.exp(
    target.mean + target.standard_deviation * output.minimumFeeZ,
  );
  if (!Number.isFinite(predictedFee) || predictedFee <= 0) {
    throw new Error("Predicted fee must be positive and finite");
  }

  return {
    selectedAction: action,
    predictedFee,
  };
}

function createInferenceResult(
  chain: Chain,
  selection: ModelSelection,
  head: Awaited<ReturnType<ChainSession["readSnapshot"]>>,
  prediction: ReturnType<typeof decodePrediction>,
): InferenceResult {
  const immediateBlock = head.number + 1n;
  const targetBlock = immediateBlock + BigInt(prediction.selectedAction);
  return {
    chain,
    K: selection.K,
    artifact_id: selection.modelManifest.artifact_id,
    head_block: safeBigInt(head.number, "head block"),
    head_hash: head.hash,
    head_base_fee_per_gas: safeBigInt(
      head.baseFeePerGas,
      "head base fee",
    ),
    selected_action_k: prediction.selectedAction,
    immediate_block: safeBigInt(immediateBlock, "immediate block"),
    target_block: safeBigInt(targetBlock, "target block"),
    predicted_minimum_base_fee_per_gas: prediction.predictedFee,
  };
}

function safeBigInt(value: bigint, label: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} exceeds the safe integer range`);
  }
  return Number(value);
}

function safeBlockInput(value: number, label: string): bigint {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a nonnegative safe integer`);
  }
  return BigInt(value);
}

function abortError(message: string): Error {
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}

function inferenceFailure(message: string, cause: unknown): Error {
  if (cause instanceof Error && cause.name === "AbortError") {
    return cause;
  }
  return new Error(message, { cause });
}
