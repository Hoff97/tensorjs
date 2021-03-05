import {Tensor} from '../../../library';
import {Mode} from '../../../model/module';
import {DType} from '../../../types';
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

  abstract calc<DTpe extends DType>(input: Tensor<DTpe>): Tensor<DTpe>;

  protected getAxes<DTpe extends DType>(input: Tensor<DTpe>) {
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

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    if (this.onnxVersion < 13) {
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
