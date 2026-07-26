import { describe, expect, it } from "vitest";
import type { Hash } from "viem";

import { buildModelInput } from "../src/features";
import type { ChainManifest, FeatureName } from "../src/features";
import type { BlockRow, FeeHistory } from "../src/rpc";
import fixture from "./fixtures/features.json";

function fixtureBlocks(): BlockRow[] {
  return fixture.blocks.map((block) => ({
    number: BigInt(block.number),
    hash: block.hash as Hash,
    parentHash: block.parentHash as Hash,
    timestamp: BigInt(block.timestamp),
    baseFeePerGas: BigInt(block.baseFeePerGas),
    gasUsed: BigInt(block.gasUsed),
    gasLimit: BigInt(block.gasLimit),
    transactionCount: block.transactionCount,
  }));
}

function fixtureFeeHistory(): FeeHistory {
  return {
    oldestBlock: BigInt(fixture.feeHistory.oldestBlock),
    baseFeePerGas: fixture.feeHistory.baseFeePerGas.map(BigInt),
    gasUsedRatio: fixture.feeHistory.gasUsedRatio,
    reward: fixture.feeHistory.reward.map((row) => row.map(BigInt)),
  };
}

function fixtureManifest(): ChainManifest {
  return {
    chain_id: fixture.manifest.chain_id,
    context_blocks: fixture.manifest.context_blocks,
    features: fixture.manifest.features.map((feature) => ({
      ...feature,
      name: feature.name as FeatureName,
    })),
  };
}

describe("buildModelInput", () => {
  it("matches the Python float32 oracle for all transforms in manifest order", () => {
    const result = buildModelInput(
      fixtureBlocks(),
      fixtureFeeHistory(),
      fixtureManifest(),
    );

    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(
      fixture.manifest.context_blocks * fixture.manifest.features.length,
    );
    result.forEach((value, index) => {
      expect(Math.abs(value - fixture.expected[index])).toBeLessThanOrEqual(1e-6);
    });
  });

  it("uses the real predecessor and exact forming-fee integer arithmetic", () => {
    const blocks = fixtureBlocks();
    const formingBlocks = [
      { ...blocks[1], baseFeePerGas: 1n, gasUsed: 101n, gasLimit: 200n },
      { ...blocks[2], baseFeePerGas: 9n, gasUsed: 0n, gasLimit: 200n },
      { ...blocks[3], baseFeePerGas: 10n, gasUsed: 100n, gasLimit: 200n },
      blocks[4],
    ];
    const intervalInput = buildModelInput(blocks, null, {
      chain_id: 1,
      context_blocks: 4,
      features: [
        {
          name: "block_interval_seconds",
          mean: 0,
          standard_deviation: 1,
        },
      ],
    });
    const formingInput = buildModelInput(formingBlocks, null, {
      chain_id: 1,
      context_blocks: 4,
      features: [
        {
          name: "log_exact_forming_base_fee_per_gas",
          mean: 0,
          standard_deviation: 1,
        },
      ],
    });

    expect(Array.from(intervalInput)).toEqual([12, 18, 15, 20]);
    ["2", "8", "10", fixture.formingChildBaseFees[3]].forEach((fee, index) => {
      expect(
        Math.abs(formingInput[index] - Math.log(Number(fee))),
      ).toBeLessThanOrEqual(1e-6);
    });
  });

  it("aligns P50 fee rows to model rows after dropping the predecessor", () => {
    const input = buildModelInput(fixtureBlocks(), fixtureFeeHistory(), {
      chain_id: 1,
      context_blocks: 4,
      features: [
        {
          name: "log1p_effective_priority_fee_per_gas_p50",
          mean: 0,
          standard_deviation: 1,
        },
        {
          name: "block_interval_seconds",
          mean: 0,
          standard_deviation: 1,
        },
      ],
    });

    expect(Array.from(input)).toEqual([
      Math.fround(Math.log1p(2_000_000_000)),
      12,
      Math.fround(Math.log1p(3_000_000_000)),
      18,
      Math.fround(Math.log1p(4_000_000_000)),
      15,
      Math.fround(Math.log1p(5_000_000_000)),
      20,
    ]);
  });

  it("rejects incomplete fee history and non-Ethereum forming fees", () => {
    const blocks = fixtureBlocks().slice(1);
    const feeHistory = fixtureFeeHistory();
    const priorityManifest: ChainManifest = {
      chain_id: 1,
      context_blocks: 4,
      features: [
        {
          name: "log1p_effective_priority_fee_per_gas_p50",
          mean: 0,
          standard_deviation: 1,
        },
      ],
    };

    expect(() =>
      buildModelInput(
        blocks,
        { ...feeHistory, oldestBlock: feeHistory.oldestBlock + 1n },
        priorityManifest,
      ),
    ).toThrow("Fee history must exactly cover model rows");
    expect(() =>
      buildModelInput(blocks, null, {
        chain_id: 137,
        context_blocks: 4,
        features: [
          {
            name: "log_exact_forming_base_fee_per_gas",
            mean: 0,
            standard_deviation: 1,
          },
        ],
      }),
    ).toThrow("Ethereum-only");
  });

  it("rejects unsafe integer conversion instead of losing raw precision", () => {
    const [block] = fixtureBlocks();

    expect(() =>
      buildModelInput(
        [{ ...block, baseFeePerGas: BigInt(Number.MAX_SAFE_INTEGER) + 1n }],
        null,
        {
          chain_id: 1,
          context_blocks: 1,
          features: [
            {
              name: "log_base_fee_per_gas",
              mean: 0,
              standard_deviation: 1,
            },
          ],
        },
      ),
    ).toThrow("baseFeePerGas exceeds the safe integer range");
  });
});
