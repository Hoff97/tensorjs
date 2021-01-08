import Long from "long";
import { onnx } from "onnx-proto";
import { Tensor } from "../library";
import { Attributes, Constants } from "./types";

export abstract class OnnxNode {
  protected attributes: {[name: string]: onnx.IAttributeProto} = {};
  protected onnxVersion: number;
  public inputs: string[];
  public outputs: string[];

  public variableInputs: number;

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

  getAttribute(name: string) {
    return this.attributes[name];
  }

  getAttributeString(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      return attr.s;
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

  async toCPU() {}
  async toWASM() {}
  async toGPU() {}

  abstract forward(inputs: Tensor[]): Tensor[];
}