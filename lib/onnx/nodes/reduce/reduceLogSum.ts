import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ReduceLogSumNode extends ReduceNode {
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
      'ReduceLogSum',
      mode
    );
  }

  calc(input: Tensor): Tensor {
    return input.reduceLogSum(this.axes, this.keepDims);
  }
}
