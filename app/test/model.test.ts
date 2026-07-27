import { describe, expect, it, vi } from "vitest";

const executorch = vi.hoisted(() => ({
  init: vi.fn(),
  module: vi.fn(),
}));

vi.mock("react-native-executorch", () => ({
  ExecutorchModule: executorch.module,
  ScalarType: { FLOAT: 6, INT: 3 },
  initExecutorch: executorch.init,
}));

vi.mock("react-native-executorch-expo-resource-fetcher", () => ({
  ExpoResourceFetcher: { name: "expo-resource-fetcher" },
}));

import {
  createModelCatalog,
  createModelRuntime,
  type MobileChainManifest,
  type MobileManifest,
  type ModelManifest,
  type ModelResourceTable,
  type ModelSelection,
} from "../src/model";
import type { Chain, Horizon } from "../src/domain";
import { flushMicrotasks } from "./helpers";

type NativeTensor = {
  dataPtr: ArrayBuffer | Float32Array;
  sizes: number[];
  scalarType: number;
};

function artifactId(index: number): string {
  return `00000000-0000-4000-8000-${index.toString().padStart(12, "0")}`;
}

function modelEntry(index: number, K: Horizon): ModelManifest {
  return {
    artifact_id: artifactId(index),
    target: { mean: K, standard_deviation: 0.5 },
  };
}

function chainManifest(firstIndex: number): MobileChainManifest {
  return {
    context_blocks: 2,
    features: [
      {
        name: "log_base_fee_per_gas" as const,
        mean: 1,
        standard_deviation: 2,
      },
    ],
    models: {
      2: modelEntry(firstIndex, 2),
      3: modelEntry(firstIndex + 1, 3),
      4: modelEntry(firstIndex + 2, 4),
      5: modelEntry(firstIndex + 3, 5),
    },
  };
}

const MANIFEST: MobileManifest = {
  chains: {
    ethereum: chainManifest(1),
    polygon: chainManifest(5),
    avalanche: chainManifest(9),
  },
};

const RESOURCES: ModelResourceTable = {
  ethereum: { 2: 12, 3: 13, 4: 14, 5: 15 },
  polygon: { 2: 22, 3: 23, 4: 24, 5: 25 },
  avalanche: { 2: 32, 3: 33, 4: 34, 5: 35 },
};

function selection(chain: Chain, K: Horizon): ModelSelection {
  return {
    chain,
    K,
    source: RESOURCES[chain][K],
    chainManifest: MANIFEST.chains[chain],
    modelManifest: MANIFEST.chains[chain].models[K],
  };
}

function output(
  values: readonly number[],
  sizes: number[],
  scalarType = 6,
): NativeTensor {
  return {
    dataPtr: new Float32Array(values).buffer,
    sizes,
    scalarType,
  };
}

function native(
  forward: (inputs: NativeTensor[]) => Promise<NativeTensor[]> = async () => [
    output([0, 1], [1, 2]),
    output([0.25], [1]),
  ],
) {
  return {
    load: vi.fn(async (_source: number) => undefined),
    forward: vi.fn(forward),
    delete: vi.fn(),
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

describe("model catalog", () => {
  it("selects the exact typed manifest and resource cell", () => {
    const catalog = createModelCatalog(MANIFEST, RESOURCES);

    expect(catalog.chainManifest("polygon")).toEqual(MANIFEST.chains.polygon);
    expect(catalog.select("polygon", 4)).toEqual({
      chain: "polygon",
      K: 4,
      source: 24,
      chainManifest: MANIFEST.chains.polygon,
      modelManifest: MANIFEST.chains.polygon.models[4],
    });
  });
});

describe("model runtime", () => {
  it("reuses an unchanged model", async () => {
    const module = native();
    const factory = vi.fn(() => module);
    const runtime = createModelRuntime(factory);
    const selected = selection("ethereum", 2);
    const input = new Float32Array([1, 2]);

    await runtime.prepare(selected);
    await runtime.prepare(selected);
    const result = await runtime.execute(selected, input);

    expect(factory).toHaveBeenCalledTimes(1);
    expect(module.load).toHaveBeenCalledOnce();
    expect(module.load).toHaveBeenCalledWith(12);
    expect(module.forward).toHaveBeenCalledWith([
      {
        dataPtr: input,
        sizes: [1, 2, 1],
        scalarType: 6,
      },
    ]);
    expect(result.actionLogits).toEqual(new Float32Array([0, 1]));
    expect(result.minimumFeeZ).toBe(0.25);

    await runtime.dispose();
    expect(module.delete).toHaveBeenCalledOnce();
  });

  it("serializes same-selection native forwards", async () => {
    const firstForward = deferred<NativeTensor[]>();
    const module = native();
    module.forward
      .mockImplementationOnce(async () => firstForward.promise)
      .mockResolvedValue([
        output([0, 1], [1, 2]),
        output([0.25], [1]),
      ]);
    const runtime = createModelRuntime(() => module);
    const selected = selection("ethereum", 2);
    const input = new Float32Array([1, 2]);

    const first = runtime.execute(selected, input);
    await vi.waitFor(() => expect(module.forward).toHaveBeenCalledOnce());
    const second = runtime.execute(selected, input);
    await flushMicrotasks();
    expect(module.forward).toHaveBeenCalledOnce();

    firstForward.resolve([
      output([0, 1], [1, 2]),
      output([0.25], [1]),
    ]);
    await expect(first).resolves.toEqual({
      actionLogits: new Float32Array([0, 1]),
      minimumFeeZ: 0.25,
    });
    await expect(second).resolves.toEqual({
      actionLogits: new Float32Array([0, 1]),
      minimumFeeZ: 0.25,
    });
    expect(module.forward).toHaveBeenCalledTimes(2);
    await runtime.dispose();
  });

  it("waits for forward before replacing the one retained model", async () => {
    const forward = deferred<NativeTensor[]>();
    const events: string[] = [];
    const first = native(async () => forward.promise);
    first.delete.mockImplementation(() => events.push("delete first"));
    const second = native();
    second.load.mockImplementation(async () => {
      events.push("load second");
    });
    const factory = vi
      .fn()
      .mockImplementationOnce(() => first)
      .mockImplementationOnce(() => second);
    const runtime = createModelRuntime(factory);
    const firstSelection = selection("ethereum", 2);
    const secondSelection = selection("ethereum", 3);

    const running = runtime
      .execute(firstSelection, new Float32Array([1, 2]))
      .catch((error: unknown) => error);
    await vi.waitFor(() => expect(first.forward).toHaveBeenCalledOnce());

    const replacing = runtime.prepare(secondSelection);
    await Promise.resolve();
    expect(first.delete).not.toHaveBeenCalled();
    expect(second.load).not.toHaveBeenCalled();

    forward.resolve([output([0, 1], [1, 2]), output([0], [1])]);
    const stale = await running;
    await replacing;

    expect(stale).toMatchObject({ name: "AbortError" });
    expect(events).toEqual(["delete first", "load second"]);
    await runtime.dispose();
  });

  it("waits for forward before disposing", async () => {
    const forward = deferred<NativeTensor[]>();
    const module = native(async () => forward.promise);
    const runtime = createModelRuntime(() => module);
    const selected = selection("ethereum", 2);

    const running = runtime
      .execute(selected, new Float32Array([1, 2]))
      .catch((error: unknown) => error);
    await vi.waitFor(() => expect(module.forward).toHaveBeenCalledOnce());
    const disposing = runtime.dispose();
    await Promise.resolve();
    expect(module.delete).not.toHaveBeenCalled();

    forward.resolve([output([0, 1], [1, 2]), output([0], [1])]);
    expect(await running).toMatchObject({ name: "AbortError" });
    await disposing;
    expect(module.delete).toHaveBeenCalledOnce();
  });

  it("retries after a model load fails", async () => {
    const failed = native();
    failed.load.mockRejectedValue(new Error("load failed"));
    const loaded = native();
    const factory = vi
      .fn()
      .mockImplementationOnce(() => failed)
      .mockImplementationOnce(() => loaded);
    const runtime = createModelRuntime(factory);
    const selected = selection("ethereum", 2);

    await expect(runtime.prepare(selected)).rejects.toThrow("load failed");
    await expect(
      runtime.execute(selected, new Float32Array([1, 2])),
    ).resolves.toEqual({
      actionLogits: new Float32Array([0, 1]),
      minimumFeeZ: 0.25,
    });
    expect(failed.delete).toHaveBeenCalledOnce();
    expect(loaded.load).toHaveBeenCalledOnce();
    await runtime.dispose();
  });

  it.each([
    {
      name: "output count",
      outputs: [output([0, 1], [1, 2])],
      message: "exactly two",
    },
    {
      name: "logit scalar type",
      outputs: [output([0, 1], [1, 2], 3), output([0], [1])],
      message: "float32",
    },
    {
      name: "logit shape",
      outputs: [output([0, 1], [2]), output([0], [1])],
      message: "[1, 2]",
    },
    {
      name: "regression shape",
      outputs: [output([0, 1], [1, 2]), output([0], [1, 1])],
      message: "[1]",
    },
    {
      name: "finite values",
      outputs: [output([0, Number.NaN], [1, 2]), output([0], [1])],
      message: "finite",
    },
  ])("rejects invalid $name", async ({ outputs, message }) => {
    const module = native(async () => outputs);
    const runtime = createModelRuntime(() => module);

    await expect(
      runtime.execute(
        selection("ethereum", 2),
        new Float32Array([1, 2]),
      ),
    ).rejects.toThrow(message);
    await runtime.dispose();
  });
});
