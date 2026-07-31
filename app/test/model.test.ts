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

import { createModelRuntime } from "../src/model";
import {
  deferred,
  flushMicrotasks,
  modelSelection,
} from "./helpers";

type NativeTensor = {
  dataPtr: ArrayBuffer | Float32Array;
  sizes: number[];
  scalarType: number;
};

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

describe("model runtime", () => {
  it("loads on first execution and reuses an unchanged model", async () => {
    const module = native();
    const factory = vi.fn(() => module);
    const runtime = createModelRuntime(factory);
    const selected = modelSelection(2);
    const input = new Float32Array([1, 2]);

    const first = await runtime.execute(selected, input);
    const second = await runtime.execute(selected, input);

    expect(factory).toHaveBeenCalledTimes(1);
    expect(module.load).toHaveBeenCalledOnce();
    expect(module.load).toHaveBeenCalledWith(12);
    expect(module.forward).toHaveBeenCalledTimes(2);
    expect(module.forward).toHaveBeenLastCalledWith([
      {
        dataPtr: input,
        sizes: [1, 2, 1],
        scalarType: 6,
      },
    ]);
    expect(first).toEqual({
      actionLogits: new Float32Array([0, 1]),
      minimumFeeZ: 0.25,
    });
    expect(second).toEqual(first);

    await runtime.dispose();
    expect(module.delete).toHaveBeenCalledOnce();
  });

  it("serializes replacement and disposal while preserving copied outputs", async () => {
    const forward = deferred<NativeTensor[]>();
    const events: string[] = [];
    const firstOutputs = [
      output([0, 1], [1, 2]),
      output([0.25], [1]),
    ];
    const first = native(async () => {
      events.push("forward first");
      return forward.promise;
    });
    first.load.mockImplementation(async () => {
      events.push("load first");
    });
    first.delete.mockImplementation(() => {
      events.push("delete first");
      for (const tensor of firstOutputs) {
        new Float32Array(tensor.dataPtr as ArrayBuffer).fill(-1);
      }
    });
    const second = native(async () => {
      events.push("forward second");
      return [output([1, 0, 2], [1, 3]), output([-0.5], [1])];
    });
    second.load.mockImplementation(async () => {
      events.push("load second");
    });
    second.delete.mockImplementation(() => events.push("delete second"));
    const factory = vi
      .fn()
      .mockImplementationOnce(() => first)
      .mockImplementationOnce(() => second);
    const runtime = createModelRuntime(factory);
    const firstSelection = modelSelection(2);
    const secondSelection = modelSelection(3);

    const firstRun = runtime.execute(
      firstSelection,
      new Float32Array([1, 2]),
    );
    await vi.waitFor(() => expect(first.forward).toHaveBeenCalledOnce());

    const secondRun = runtime.execute(
      secondSelection,
      new Float32Array([1, 2]),
    );
    const disposal = runtime.dispose();
    await flushMicrotasks();
    expect(first.delete).not.toHaveBeenCalled();
    expect(second.load).not.toHaveBeenCalled();

    forward.resolve(firstOutputs);
    await expect(firstRun).resolves.toEqual({
      actionLogits: new Float32Array([0, 1]),
      minimumFeeZ: 0.25,
    });
    await expect(secondRun).resolves.toEqual({
      actionLogits: new Float32Array([1, 0, 2]),
      minimumFeeZ: -0.5,
    });
    await disposal;
    expect(events).toEqual([
      "load first",
      "forward first",
      "delete first",
      "load second",
      "forward second",
      "delete second",
    ]);
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
      message: "shape [1, 2]",
    },
    {
      name: "logit storage length",
      outputs: [output([0], [1, 2]), output([0], [1])],
      message: "exactly 2",
    },
    {
      name: "minimum shape",
      outputs: [output([0, 1], [1, 2]), output([0], [1, 1])],
      message: "shape [1]",
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
        modelSelection(2),
        new Float32Array([1, 2]),
      ),
    ).rejects.toThrow(message);
    await runtime.dispose();
  });
});
