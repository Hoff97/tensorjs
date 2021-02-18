import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ReduceL1Node extends ReduceNode {
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
      'ReduceL1',
      mode
    );
  }

  calc(input: Tensor): Tensor {
    const abs = input.abs();
    const result = abs.sum(this.axes, this.keepDims);
    abs.delete();
    return result;
  }
}
