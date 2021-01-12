import { CPUTensor } from "../../tensor/cpu/tensor";
import { glContext } from "../../tensor/gpu/gl";
import Tensor from "../../types";
import { toCPU, toGPU, toWASM } from "../../util/convert";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class InstanceNormalizationNode extends OnnxNode {
  private epsilon: number;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.epsilon = this.getAttributeFloat('epsilon') || 1e-05;

    //TODO: Handle onnx versions < 6 here
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    let scale = inputs[1];
    let B = inputs[2];

    const dataRank = x.getShape().length - 2;

    const C = scale.getShape()[0];

    const newShape = [1,C,...new Array(dataRank).fill(1)];

    scale = scale.reshape(newShape, false);
    B = B.reshape(newShape, false);

    const reduceAxes = new Array(x.getShape().length  -2);
    for (let i = 0; i < dataRank; i++) {
      reduceAxes[i] = i+2;
    }

    const mean = x.reduceMean(reduceAxes, true);
    glContext.flush();
    const diff = x.subtract(mean);
    glContext.flush();
    const variance = diff.reduceMeanSquare(reduceAxes, true);
    glContext.flush();

    const result = x.normalize(mean, variance, this.epsilon, scale, B);

    mean.delete();
    diff.delete();
    variance.delete();

    return [result];
  }
}