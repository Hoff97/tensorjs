import Tensor from '../../types';
import {OnnxNode} from '../node';
import {ConvNode} from '../nodes/conv';
import {ReluNode} from '../nodes/relu';
import {Constants} from '../types';
import {SequenceOptimization} from './optimization';

export class ConvRelu extends SequenceOptimization {
  constructor() {
    super(['Conv', 'Relu']);
  }

  apply(
    nodes: OnnxNode[],
    resolveConstant: (name: string) => Tensor,
    constants: Constants,
    onnxVersion: number
  ): OnnxNode {
    const conv = nodes[0] as ConvNode;
    const relu = nodes[1] as ReluNode;

    return new ConvNode(
      Object.entries(conv.attributes).map(x => x[1]),
      conv.inputs,
      relu.outputs,
      constants,
      onnxVersion,
      conv.kernel,
      conv.bias,
      'relu'
    );
  }
}
