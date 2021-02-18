import {Tensor} from '../../../library';
import {Mode} from '../../../model/module';
import {OnnxNode} from '../../node';
import {Attributes, Constants} from '../../types';

export abstract class ReduceNode extends OnnxNode {
  protected axes?: number[];
  protected keepDims?: boolean;

  protected name: string;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    name: string,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.axes = this.getAttributeInts('axes');
    const keep = this.getAttributeInt('keepdims');

    this.keepDims = keep === 1 || keep === undefined;

    this.name = name;
  }

  abstract calc(input: Tensor): Tensor;

  protected getAxes(input: Tensor) {
    if (this.axes !== undefined) {
      return this.axes;
    } else {
      const rank = input.getShape().length;

      const res = new Array(rank);
      for (let i = 0; i < rank; i++) {
        res[i] = i;
      }
      return res;
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      return [this.calc(inputs[0])];
    }
    throw new Error(
      `${this.name} is not implemented for onnx version ${this.onnxVersion}`
    );
  }

  getType() {
    return this.name;
  }

  delete(): void {}
}
