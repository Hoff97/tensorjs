import Long from "long";
import { onnx } from "onnx-proto";
import { Tensor } from "../library";
import { PrototypeTensor } from "../tensor/cpu/prototype";
import { CPUTensor } from "../tensor/cpu/tensor";
import { defaultAllocator } from "../tensor/gpu/gl";
import { GPUMemoryAllocator, MemoryEntry } from "../tensor/gpu/memory";
import { GPUTensor } from "../tensor/gpu/tensor";
import { Precision } from "../types";
import { Attributes, Constants } from "./types";

export abstract class OnnxNode {
  protected onnxVersion: number;

  protected allocator = defaultAllocator;

  public inputs: string[];
  public outputs: string[];

  public variableInputs: number;
  public attributes: {[name: string]: onnx.IAttributeProto} = {};

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    for (let i = 0; i < attributes.length; i++) {
      this.attributes[attributes[i].name] = attributes[i];
    }
    this.inputs = inputs;
    this.outputs = outputs;

    this.onnxVersion = onnxVersion;

    this.variableInputs = 0;
    for (let i = 0; i < this.inputs.length; i++) {
      if (constants[this.inputs[i]] === undefined) {
        this.variableInputs++;
      }
    }
  }

  initialize(resolveConstant: (name: string) => Tensor) {}

  getAttribute(name: string) {
    return this.attributes[name];
  }

  getAttributeString(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      const str = attr.s;
      if (str !== undefined) {
        return new TextDecoder("utf-8").decode(str);
      }
      return undefined;
    }
    return undefined;
  }

  getAttributeInts(name: string): number[] {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      const result = this.attributes[name].ints;
      for (let i = 0; i < result.length; i++) {
        if (Long.isLong(result[i])) {
          result[i] = (result[i] as Long).toNumber();
        }
      }
      return result as any;
    }
    return undefined;
  }

  getAttributeInt(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      let result = attr.i;
      if (Long.isLong(result)) {
        result = (result as Long).toNumber();
      }
      return result;
    }
    return undefined;
  }

  getAttributeFloat(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      let result = attr.f;
      return result;
    }
    return undefined;
  }

  getAttributeTensor(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      let result = attr.t;
      return result;
    }
    return undefined;
  }

  setAllocator(allocator: GPUMemoryAllocator) {
    this.allocator = allocator;
  }

  async toCPU() {}
  async toWASM() {}
  async toGPU(precision: Precision) {}

  abstract forward(inputs: Tensor[]): Promise<Tensor[]>;

  allStaticCPU(inputs: Tensor[]) {
    return inputs.every(x => x instanceof CPUTensor && x.values !== null);
  }

  getMemoryEntries(inputs: Tensor[]): MemoryEntry[] {
    const res = inputs.every(x => (x instanceof PrototypeTensor && x.memory !== undefined) || (x instanceof GPUTensor && x.memory !== undefined));
    if (!res) {
      throw new Error('Not all tensors are tensors with gpu memory attached')
    }
    return inputs.map(x => (x as any).memory);
  }

  async defaultStaticForward(inputs: Tensor[]): Promise<{outputs: CPUTensor[]}> {
    const outputs = await this.forward(inputs);
    return {outputs: outputs as CPUTensor[]};
  }

  abstract staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{outputs: (CPUTensor | PrototypeTensor)[]}>;

  abstract initializeForCompiling(): void;

  abstract getType(): string;

  abstract delete(): void;
}