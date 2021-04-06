import {Mode} from '../../../model/module';
import {poolResultShape} from '../../../ops/util/pool';
import Tensor, {DType} from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ArgMinNode extends ReduceNode {
  protected selectLastIndex: boolean;
  protected axis: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'ArgMin', mode);

    this.selectLastIndex = this.getAttributeInt('select_last_index') === 1;

    const ax = this.getAttributeInt('axis');
    if (ax !== undefined && ax !== null) {
      this.axis = ax;
    } else {
      this.axis = 0;
    }
  }

  calc<DTpe extends DType>(input: Tensor<DTpe>): Tensor<DTpe> {
    const result = input.argMin([this.axis], this.selectLastIndex);

    const inputShape = input.getShape();

    const [resultShape, _] = poolResultShape(
      inputShape,
      [this.axis < 0 ? this.axis + inputShape.length : this.axis],
      this.keepDims === false ? false : true
    );

    return result.reshape(resultShape, false) as any;
  }
}
