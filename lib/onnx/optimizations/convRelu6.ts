import Tensor from '../../types';
import {OnnxNode} from '../node';
import {ClipNode} from '../nodes/clip';
import {ConvNode} from '../nodes/conv';
import {Constants} from '../types';
import {SequenceOptimization} from './optimization';

export class ConvRelu6 extends SequenceOptimization {
  constructor() {
    super(['Conv', 'Clip']);
  }

  apply(
    nodes: OnnxNode[],
    resolveConstant: (name: string) => Tensor,
    constants: Constants,
    onnxVersion: number
  ): OnnxNode {
    const conv = nodes[0] as ConvNode;
    const clip = nodes[1] as ClipNode;

    return new ConvNode(
      Object.entries(conv.attributes).map(x => x[1]),
      conv.inputs,
      clip.outputs,
      constants,
      onnxVersion,
      conv.kernel,
      conv.bias,
      'relu6'
    );
  }

  canApply(nodes: OnnxNode[]) {
    const clip = nodes[1] as ClipNode;
    return clip.min === 0 && clip.max === 6;
  }
}
