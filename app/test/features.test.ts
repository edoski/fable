import { describe, expect, it } from "vitest";
import type { Hash } from "viem";

import type { BlockRow } from "../src/domain";
import { buildModelInput } from "../src/features";
import type { ChainManifest, FeatureName } from "../src/features";
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

function fixturePriorityFeeRewards(): readonly (readonly [bigint, bigint])[] {
  return fixture.feeHistory.reward.map(([p50, p90]) => [
    BigInt(p50),
    BigInt(p90),
  ]);
}

function fixtureManifest(): ChainManifest {
  return {
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
      fixturePriorityFeeRewards(),
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

  it("uses exact forming-fee integer arithmetic", () => {
    const blocks = fixtureBlocks();
    const formingBlocks = [
      { ...blocks[1], baseFeePerGas: 1n, gasUsed: 101n, gasLimit: 200n },
      { ...blocks[2], baseFeePerGas: 9n, gasUsed: 0n, gasLimit: 200n },
      { ...blocks[3], baseFeePerGas: 10n, gasUsed: 100n, gasLimit: 200n },
      blocks[4],
    ];
    const formingInput = buildModelInput(formingBlocks, null, {
      context_blocks: 4,
      features: [
        {
          name: "log_exact_forming_base_fee_per_gas",
          mean: 0,
          standard_deviation: 1,
        },
      ],
    });

    ["2", "8", "10", fixture.formingChildBaseFees[3]].forEach((fee, index) => {
      expect(
        Math.abs(formingInput[index] - Math.log(Number(fee))),
      ).toBeLessThanOrEqual(1e-6);
    });
  });

  it("rejects nonfinite final float32 features", () => {
    expect(() =>
      buildModelInput(fixtureBlocks().slice(1, 2), null, {
        context_blocks: 1,
        features: [
          {
            name: "log_base_fee_per_gas",
            mean: 0,
            standard_deviation: 0,
          },
        ],
      }),
    ).toThrow("Model input must contain finite float32 values");
  });

  it("requires complete nonnegative priority-fee rewards", () => {
    const blocks = fixtureBlocks();
    const manifest = fixtureManifest();
    const rewards = fixturePriorityFeeRewards();
    const missingP50 = rewards.map(
      (reward) => [...reward] as [bigint, bigint],
    );
    const missingP90 = rewards.map(
      (reward) => [...reward] as [bigint, bigint],
    );
    Reflect.deleteProperty(missingP50[0], "0");
    Reflect.deleteProperty(missingP90[0], "1");
    const malformed = [
      null,
      rewards.slice(0, -1),
      missingP50,
      missingP90,
      rewards.map((reward, index) =>
        index === 0 ? ([-1n, reward[1]] as const) : reward,
      ),
    ];

    for (const values of malformed) {
      expect(() =>
        buildModelInput(blocks, values, manifest),
      ).toThrow(
        "Priority-fee rewards must provide nonnegative P50 and P90 values for every context block",
      );
    }
  });
});
