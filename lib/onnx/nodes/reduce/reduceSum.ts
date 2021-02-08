import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ReduceSumNode extends ReduceNode {
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
      'ReduceSum',
      mode
    );
  }

  calc(input: Tensor): Tensor {
    return input.sum(this.axes, this.keepDims);
  }
}
