import {Mode} from '../../model/module';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class SliceNode extends OnnxNode {
  private axes?: number[];
  private starts?: number[];
  private ends?: number[];

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
    this.starts = this.getAttributeInts('starts');
    this.ends = this.getAttributeInts('ends');
  }

  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    if (this.onnxVersion < 10) {
      if (this.starts === undefined || this.ends === undefined) {
        throw new Error(
          'Slice with onnx version < 10 needs starts and ends defined as attributes'
        );
      }
      const x = inputs[0];
      return [x.slice(this.starts, this.ends, this.axes)];
    } else {
      const x = inputs[0];
      const starts = inputs[1];
      const ends = inputs[2];
      const axes = inputs[3];
      const steps = inputs[4];

      const startValues = await this.toValues(starts);
      const endValues = await this.toValues(ends);
      const axesValues =
        axes !== undefined ? await this.toValues(axes) : undefined;
      const stepValues =
        steps !== undefined ? await this.toValues(steps) : undefined;

      return [x.slice(startValues, endValues, axesValues, stepValues)];
    }
  }

  getType() {
    return 'Slice';
  }

  delete(): void {}
}
