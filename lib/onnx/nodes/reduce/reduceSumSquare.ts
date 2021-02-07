import {Mode} from '../../../model/module';
import types from '../../../types';
import {Attributes, Constants} from '../../types';
import {ReduceNode} from './reduceNode';

export class ReduceSumSquareNode extends ReduceNode {
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
      'ReduceSumSquare',
      mode
    );
  }

  calc(input: types): types {
    return input.sumSquare(this.axes, this.keepDims);
  }
}
