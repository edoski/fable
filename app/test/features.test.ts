import { describe, expect, it } from "vitest";
import type { Hash } from "viem";

import { buildModelInput } from "../src/features";
import type { ChainManifest, FeatureName } from "../src/features";
import type { BlockRow } from "../src/rpc";
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

  it("accepts same-second blocks as a zero interval", () => {
    const blocks = fixtureBlocks();
    blocks[2] = { ...blocks[2], timestamp: blocks[1].timestamp };

    expect(
      buildModelInput(blocks.slice(1, 3), null, {
        context_blocks: 1,
        features: [
          {
            name: "block_interval_seconds",
            mean: 0,
            standard_deviation: 1,
          },
        ],
      }),
    ).toEqual(new Float32Array([0]));
  });

  it("rejects unsafe integer conversion instead of losing raw precision", () => {
    const [block] = fixtureBlocks();

    expect(() =>
      buildModelInput(
        [{ ...block, baseFeePerGas: BigInt(Number.MAX_SAFE_INTEGER) + 1n }],
        null,
        {
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
