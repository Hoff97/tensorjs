import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ReduceMeanNode extends ReduceNode {
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
      'ReduceMean',
      mode
    );
  }

  calc(input: Tensor): Tensor {
    return input.reduceMean(this.axes, this.keepDims);
  }
}
