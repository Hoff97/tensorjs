import {Mode} from '../../model/module';
import Tensor, {DType} from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class UnsqueezeNode extends OnnxNode {
  private axes?: number[];

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    if (onnxVersion < 13) {
      this.axes = this.getAttributeInts('axes');
    }
  }

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    const x = inputs[0];

    if (this.onnxVersion < 13 && this.axes !== undefined) {
      const currShape = x.getShape();
      const newShape = [];
      let axIx = 0;
      for (let i = 0; i < currShape.length; i++) {
        if (axIx < this.axes.length && this.axes[axIx] === i) {
          newShape.push(1);
          axIx++;
        }
        newShape.push(currShape[i]);
      }
      if (this.axes[this.axes.length - 1] === currShape.length) {
        newShape.push(1);
      }

      return [x.reshape(newShape)];
    }
    throw new Error(
      `Unsqueeze with onnx version ${this.onnxVersion} not yet implemented`
    );
  }

  getType() {
    return 'Unsqueeze';
  }

  delete(): void {}
}
