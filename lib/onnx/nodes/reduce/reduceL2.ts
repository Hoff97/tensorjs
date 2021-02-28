import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ReduceL2Node extends ReduceNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      'ReduceL2',
      mode
    );
  }

  calc<DTpe extends DType>(input: Tensor<DTpe>): Tensor<DTpe> {
    const sumSquare = input.sumSquare(this.axes, this.keepDims);
    const result = sumSquare.sqrt();
    sumSquare.delete();
    return result;
  }
}
