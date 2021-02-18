import {Mode} from '../../../model/module';
import types from '../../../types';
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

  calc(input: types): types {
    const sumSquare = input.sumSquare(this.axes, this.keepDims);
    const result = sumSquare.sqrt();
    sumSquare.delete();
    return result;
  }
}
