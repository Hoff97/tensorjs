import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ReduceLogSumExpNode extends ReduceNode {
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
      'ReduceLogSumExp',
      mode
    );
  }

  calc<DTpe extends DType>(input: Tensor<DTpe>): Tensor<DTpe> {
    return input.reduceLogSumExp(this.axes, this.keepDims);
  }
}
