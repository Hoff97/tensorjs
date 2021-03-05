import Tensor from '../../types';
import {OnnxNode} from '../node';
import {BatchNormalizationNode} from '../nodes/batchNormalization';
import {ConvNode} from '../nodes/conv/conv';
import {Constants} from '../types';
import {SequenceOptimization} from './optimization';

export class ConvBatchNorm extends SequenceOptimization {
  constructor() {
    super(['Conv', 'BatchNormalization']);
  }

  apply(
    nodes: OnnxNode[],
    resolveConstant: (name: string) => Tensor<any>,
    constants: Constants,
    onnxVersion: number
  ): OnnxNode {
    const conv = nodes[0] as ConvNode;
    const batchNorm = nodes[1] as BatchNormalizationNode;

    const kernelConv = resolveConstant(conv.inputs[1]);
    const biasConv = resolveConstant(conv.inputs[2]);

    const scaleBN = resolveConstant(batchNorm.inputs[1]);
    const biasBN = resolveConstant(batchNorm.inputs[2]);
    const meanBN = resolveConstant(batchNorm.inputs[3]);
    const varianceBN = resolveConstant(batchNorm.inputs[4]);

    const varSqrt = varianceBN.add(batchNorm.epsTensor).sqrt();

    const scale = scaleBN.divide(varSqrt);
    varSqrt.delete();
    const bias = biasBN.subtract(meanBN.multiply(scale));

    const newShape = [
      ...scale.getShape(),
      ...new Array(kernelConv.getShape().length - scale.getShape().length).fill(
        1
      ),
    ];

    const newKernel = kernelConv.multiply(scale.reshape(newShape, false));
    let newBias = bias;
    if (biasConv !== undefined) {
      const scaledBias = biasConv.multiply(scale);
      newBias = newBias.add(scaledBias);
      scaledBias.delete();
    }

    return new ConvNode(
      Object.entries(conv.attributes).map(x => x[1]),
      [conv.inputs[0]],
      batchNorm.outputs,
      constants,
      onnxVersion,
      conv.mode,
      newKernel,
      newBias
    );
  }
}
