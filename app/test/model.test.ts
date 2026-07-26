import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

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
  type MobileManifest,
  type ModelResourceTable,
} from "../src/model";

type NativeTensor = {
  dataPtr: ArrayBuffer | Float32Array;
  sizes: number[];
  scalarType: number;
};

function artifactId(index: number): string {
  return `00000000-0000-4000-8000-${index.toString().padStart(12, "0")}`;
}

function bundle(): {
  manifest: MobileManifest;
  resources: ModelResourceTable;
} {
  let cell = 1;
  const chain = (chainId: number) => ({
    chain_id: chainId,
    context_blocks: 2,
    features: [
      {
        name: "log_base_fee_per_gas" as const,
        mean: 1,
        standard_deviation: 2,
      },
    ],
    models: {
      2: {
        artifact_id: artifactId(cell++),
        target: { mean: 2, standard_deviation: 0.5 },
      },
      3: {
        artifact_id: artifactId(cell++),
        target: { mean: 3, standard_deviation: 0.5 },
      },
      4: {
        artifact_id: artifactId(cell++),
        target: { mean: 4, standard_deviation: 0.5 },
      },
      5: {
        artifact_id: artifactId(cell++),
        target: { mean: 5, standard_deviation: 0.5 },
      },
    },
  });
  return {
    manifest: {
      executorch_version: "1.2.0",
      chains: {
        ethereum: chain(1),
        polygon: chain(137),
        avalanche: chain(43_114),
      },
    },
    resources: {
      ethereum: { 2: 12, 3: 13, 4: 14, 5: 15 },
      polygon: { 2: 22, 3: 23, 4: 24, 5: 25 },
      avalanche: { 2: 32, 3: 33, 4: 34, 5: 35 },
    },
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

async function flushMicrotasks(): Promise<void> {
  for (let index = 0; index < 10; index += 1) {
    await Promise.resolve();
  }
}

describe("bundled model catalog", () => {
  it("owns exactly one manifest and the twelve final static Metro assets", () => {
    const source = readFileSync(
      fileURLToPath(new URL("../src/model.ts", import.meta.url).href),
      "utf8",
    );
    const expectedModels = ["ethereum", "polygon", "avalanche"].flatMap(
      (chain) =>
        [2, 3, 4, 5].map(
          (K) => `../assets/models/${chain}-k${K}.pte`,
        ),
    );
    const modelRequires = [
      ...source.matchAll(
        /require\("(\.\.\/assets\/models\/[^"]+\.pte)"\)/g,
      ),
    ].map((match) => match[1]);
    const manifestRequires = [
      ...source.matchAll(
        /require\("(\.\.\/assets\/models\/manifest\.json)"\)/g,
      ),
    ].map((match) => match[1]);

    expect(modelRequires).toEqual(expectedModels);
    expect(manifestRequires).toEqual([
      "../assets/models/manifest.json",
    ]);
  });
});

describe("model catalog", () => {
  it("validates the exporter contract and selects the exact cell", () => {
    const { manifest, resources } = bundle();
    const catalog = createModelCatalog(manifest, resources);

    expect(catalog.chainManifest("polygon")).toEqual(manifest.chains.polygon);
    expect(catalog.select("polygon", 4)).toEqual({
      chain: "polygon",
      K: 4,
      source: 24,
      chainManifest: manifest.chains.polygon,
      modelManifest: manifest.chains.polygon.models[4],
    });
  });

  it("rejects incomplete, mismatched, or non-static catalogs", () => {
    const invalidManifests: unknown[] = [];

    const extra = structuredClone(bundle().manifest) as MobileManifest & {
      extra?: boolean;
    };
    extra.extra = true;
    invalidManifests.push(extra);

    const missingModel = structuredClone(bundle().manifest);
    delete (missingModel.chains.ethereum.models as Partial<
      typeof missingModel.chains.ethereum.models
    >)[5];
    invalidManifests.push(missingModel);

    const wrongChain = structuredClone(bundle().manifest);
    wrongChain.chains.polygon.chain_id = 1;
    invalidManifests.push(wrongChain);

    const invalidTarget = structuredClone(bundle().manifest);
    invalidTarget.chains.avalanche.models[3].target.standard_deviation = 0;
    invalidManifests.push(invalidTarget);

    for (const manifest of invalidManifests) {
      expect(() =>
        createModelCatalog(manifest, bundle().resources),
      ).toThrow();
    }

    const resources = structuredClone(bundle().resources) as Record<
      string,
      Record<number, number | string>
    >;
    resources.ethereum[2] = "https://models.invalid/ethereum-k2.pte";
    expect(() =>
      createModelCatalog(bundle().manifest, resources),
    ).toThrow("static Metro asset");
  });
});

describe("model runtime", () => {
  it("initializes the Expo fetcher once and reuses an unchanged model", async () => {
    const { manifest, resources } = bundle();
    const catalog = createModelCatalog(manifest, resources);
    const module = native();
    const factory = vi.fn(() => module);
    const runtime = createModelRuntime(factory);
    const selection = catalog.select("ethereum", 2);
    const input = new Float32Array([1, 2]);

    await runtime.prepare(selection);
    await runtime.prepare(selection);
    const result = await runtime.execute(selection, input);

    expect(executorch.init).toHaveBeenCalledTimes(1);
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
    const { manifest, resources } = bundle();
    const catalog = createModelCatalog(manifest, resources);
    const firstForward = deferred<NativeTensor[]>();
    const module = native();
    module.forward
      .mockImplementationOnce(async () => firstForward.promise)
      .mockResolvedValue([
        output([0, 1], [1, 2]),
        output([0.25], [1]),
      ]);
    const runtime = createModelRuntime(() => module);
    const selection = catalog.select("ethereum", 2);
    const input = new Float32Array([1, 2]);

    const first = runtime.execute(selection, input);
    await vi.waitFor(() => expect(module.forward).toHaveBeenCalledOnce());
    const second = runtime.execute(selection, input);
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
    const { manifest, resources } = bundle();
    const catalog = createModelCatalog(manifest, resources);
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
    const firstSelection = catalog.select("ethereum", 2);
    const secondSelection = catalog.select("ethereum", 3);

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
    const { manifest, resources } = bundle();
    const catalog = createModelCatalog(manifest, resources);
    const forward = deferred<NativeTensor[]>();
    const module = native(async () => forward.promise);
    const runtime = createModelRuntime(() => module);
    const selection = catalog.select("ethereum", 2);

    const running = runtime
      .execute(selection, new Float32Array([1, 2]))
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
    const { manifest, resources } = bundle();
    const catalog = createModelCatalog(manifest, resources);
    const failed = native();
    failed.load.mockRejectedValue(new Error("load failed"));
    const loaded = native();
    const factory = vi
      .fn()
      .mockImplementationOnce(() => failed)
      .mockImplementationOnce(() => loaded);
    const runtime = createModelRuntime(factory);
    const selection = catalog.select("ethereum", 2);

    await expect(runtime.prepare(selection)).rejects.toThrow("load failed");
    await expect(
      runtime.execute(selection, new Float32Array([1, 2])),
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
    const { manifest, resources } = bundle();
    const catalog = createModelCatalog(manifest, resources);
    const module = native(async () => outputs);
    const runtime = createModelRuntime(() => module);

    await expect(
      runtime.execute(
        catalog.select("ethereum", 2),
        new Float32Array([1, 2]),
      ),
    ).rejects.toThrow(message);
    await runtime.dispose();
  });

  it("rejects an invalid row-major input before native execution", async () => {
    const { manifest, resources } = bundle();
    const catalog = createModelCatalog(manifest, resources);
    const module = native();
    const runtime = createModelRuntime(() => module);
    const selection = catalog.select("ethereum", 2);

    await expect(
      runtime.execute(selection, new Float32Array([1])),
    ).rejects.toThrow("exactly 2");
    await expect(
      runtime.execute(
        selection,
        new Float32Array([1, Number.POSITIVE_INFINITY]),
      ),
    ).rejects.toThrow("finite");
    expect(module.forward).not.toHaveBeenCalled();
    await runtime.dispose();
  });
});
