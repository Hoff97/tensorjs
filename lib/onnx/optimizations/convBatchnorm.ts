import Tensor from "../../types";
import { OnnxNode } from "../node";
import { BatchNormalizationNode } from "../nodes/batchNormalization";
import { ConvNode } from "../nodes/conv";
import { SequenceOptimization } from "./optimization";

export class ConvBatchNorm extends SequenceOptimization {
  constructor() {
    super(['Conv', 'BatchNormalization']);
  }

  apply(nodes: OnnxNode[], resolveConstant: (name: string) => Tensor): OnnxNode {
    const conv = nodes[0] as ConvNode;
    const batchNorm = nodes[1] as BatchNormalizationNode;

    const w = resolveConstant(conv.inputs[1]);
    const b = resolveConstant(conv.inputs[2]);

    const scale = batchNorm.scale;
    const bias = batchNorm.bias;

    console.log(w.getShape(), b?.getShape(), scale.getShape(), bias.getShape());

    console.log(scale, bias);

    throw new Error("Method not implemented.");
  }
}