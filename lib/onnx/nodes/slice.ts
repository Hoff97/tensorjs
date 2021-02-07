import {Mode} from '../../model/module';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class SliceNode extends OnnxNode {
  private axes?: number[];
  private starts: number[];
  private ends: number[];

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.axes = this.getAttributeInts('axes');
    //@ts-ignore
    this.starts = this.getAttributeInts('starts');
    //@ts-ignore
    this.ends = this.getAttributeInts('ends');
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      const x = inputs[0];
      return [x.slice(this.starts, this.ends, this.axes)];
    }
    throw new Error(
      `Slice not implemented for onnx version ${this.onnxVersion}`
    );
  }

  getType() {
    return 'Slice';
  }

  delete(): void {}
}
