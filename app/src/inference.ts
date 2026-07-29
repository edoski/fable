import { buildModelInput } from "./features";
import type { BlockRow, Chain, Horizon } from "./domain";
import { createDefaultModelCatalog, createModelRuntime } from "./model";
import type {
  ModelCatalog,
  ModelOutput,
  ModelRuntime,
  ModelSelection,
} from "./model";
import { createChainSession } from "./rpc";
import type { ChainSession } from "./rpc";

export type InferenceResult = {
  chain: Chain;
  K: Horizon;
  artifact_id: string;
  head_block: number;
  head_hash: string;
  selected_action_k: number;
  target_block: number;
  predicted_minimum_base_fee_per_gas: number;
};

export type ChainSnapshot = {
  head_block: number;
  current_base_fee_per_gas: number;
};

export type InferenceOutcome = {
  immediate_base_fee_per_gas: number;
  selected_base_fee_per_gas: number;
};

export type InferenceEngine = {
  watchBlocks(
    onSnapshot: (snapshot: ChainSnapshot) => void,
    onError?: (error: unknown) => void,
  ): void;
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
  async function run(K: Horizon): Promise<InferenceResult> {
    const selection = catalog.select(chain, K);
    const context = await attempt("Could not read the selected chain.", () =>
      session.sync(),
    );

    const head = context.blocks[context.blocks.length - 1];
    const input = await attempt(
      "Chain data is incomplete or invalid.",
      async () =>
        buildModelInput(
          context.blocks,
          context.priorityFeeRewards,
          selection.chainManifest,
        ),
    );
    const prediction = await attempt(
      "Could not run the selected model.",
      async () =>
        decodePrediction(
          selection,
          await model.execute(selection, input),
        ),
    );
    return attempt(
      "Chain data is incomplete or invalid.",
      async () =>
        createInferenceResult(
          chain,
          selection,
          head,
          prediction,
        ),
    );
  }

  function watchBlocks(
    onSnapshot: (snapshot: ChainSnapshot) => void,
    onError?: (error: unknown) => void,
  ): void {
    session.watchBlocks((block) => {
      onSnapshot({
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
    const immediate = BigInt(immediateBlock);
    const selected = BigInt(selectedBlock);
    const outcome = await session.readOutcome(immediate, selected);
    return {
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
    session.dispose();
    return model.dispose();
  }

  return {
    watchBlocks,
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
  let action = 0;
  for (let index = 0; index < output.actionLogits.length; index += 1) {
    if (output.actionLogits[index] > output.actionLogits[action]) {
      action = index;
    }
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
  head: BlockRow,
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
    selected_action_k: prediction.selectedAction,
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

async function attempt<T>(
  message: string,
  work: () => T | Promise<T>,
): Promise<T> {
  try {
    return await work();
  } catch (error) {
    throw inferenceFailure(message, error);
  }
}

function inferenceFailure(message: string, cause: unknown): Error {
  return new Error(message, { cause });
}
