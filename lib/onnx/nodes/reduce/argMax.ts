import {Mode} from '../../../model/module';
import {poolResultShape} from '../../../ops/util/pool';
import Tensor, {DType} from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ArgMaxNode extends ReduceNode {
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
    super(attributes, inputs, outputs, constants, onnxVersion, 'ArgMax', mode);

    this.selectLastIndex = this.getAttributeInt('select_last_index') === 1;
    this.axis = this.getAttributeInt('axis') || 0;
  }

  calc<DTpe extends DType>(input: Tensor<DTpe>): Tensor<DTpe> {
    const result = input.argMax([this.axis], this.selectLastIndex);

    const [resultShape, _] = poolResultShape(
      input.getShape(),
      [this.axis],
      this.keepDims === false ? false : true
    );

    return result.reshape(resultShape, false) as any;
  }
}
