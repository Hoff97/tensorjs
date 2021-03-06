// eslint-disable-next-line node/no-extraneous-import
import Long from 'long';
import {onnx} from 'onnx-proto';
import {Tensor} from '../library';
import {Mode, Module} from '../model/module';
import {CPUTensor} from '../tensor/cpu/tensor';
import {DType} from '../types';
import {toCPU} from '../util/convert';
import {Attributes, Constants} from './types';

export abstract class OnnxNode extends Module {
  protected onnxVersion: number;

  public inputs: string[];
  public outputs: string[];

  public variableInputs: number;
  public attributes: {[name: string]: onnx.IAttributeProto} = {};

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super();

    this.mode = mode;

    for (let i = 0; i < attributes.length; i++) {
      this.attributes[attributes[i].name as string] = attributes[i];
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

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  initialize(resolveConstant: (name: string) => Tensor<any> | undefined) {}

  getAttribute(name: string) {
    return this.attributes[name];
  }

  getAttributeString(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      const str = attr.s;
      if (str !== undefined && str !== null) {
        // eslint-disable-next-line node/no-unsupported-features/node-builtins
        return new TextDecoder('utf-8').decode(str);
      }
      return undefined;
    }
    return undefined;
  }

  getAttributeInts(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      const result = this.attributes[name].ints;
      if (result !== undefined && result !== null) {
        for (let i = 0; i < result.length; i++) {
          if (Long.isLong(result[i])) {
            result[i] = (result[i] as Long).toNumber();
          }
        }
        return result as number[];
      }
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
      const result = attr.f;
      return result;
    }
    return undefined;
  }

  getAttributeFloats(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      const result = attr.floats;
      return result;
    }
    return undefined;
  }

  getAttributeTensor(name: string) {
    const attr = this.attributes[name];
    if (attr !== undefined) {
      const result = attr.t;
      return result;
    }
    return undefined;
  }

  async toValues<DTpe extends DType>(tensor: Tensor<DTpe>): Promise<number[]> {
    if (!(tensor instanceof CPUTensor)) {
      console.warn('Tensor for values not on CPU, need to transfer!');
      tensor = await toCPU(tensor);
    }

    const sc = tensor as CPUTensor<DTpe>;

    const values = new Array(sc.size);
    for (let i = 0; i < sc.size; i++) {
      values[i] = sc.get(i);
    }
    return values;
  }

  async toCPU() {}
  async toWASM() {}
  async toGPU() {}

  abstract forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]>;

  abstract getType(): string;

  abstract delete(): void;
}
