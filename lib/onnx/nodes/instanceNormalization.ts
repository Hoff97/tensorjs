import { CPUTensor } from "../../tensor/cpu/tensor";
import { glContext } from "../../tensor/gpu/gl";
import Tensor from "../../types";
import { toCPU, toGPU, toWASM } from "../../util/convert";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class InstanceNormalizationNode extends OnnxNode {
  private epsilon: number;
  private momentum: number;

  private epsTensor: Tensor;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.epsilon = this.getAttributeFloat('epsilon') || 1e-05;

    this.epsTensor = new CPUTensor([1], [this.epsilon]);

    //TODO: Handle onnx versions < 6 here
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    let scale = inputs[1];
    let B = inputs[2];

    const dataRank = x.getShape().length - 2;
    //TODO: Handle lower onnx versions here

    const C = scale.getShape()[0];

    const newShape = [1,C,...new Array(dataRank).fill(1)];

    scale = scale.reshape(newShape);
    B = B.reshape(newShape);

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

    const varEps = variance.add(this.epsTensor);
    glContext.flush();
    const varEpsSqrt = varEps.sqrt();
    const xmean = x.subtract(mean);
    glContext.flush();
    const normalized = xmean.divide(varEpsSqrt);

    glContext.flush();
    const scaled = normalized.multiply(scale);
    glContext.flush();
    const result = scaled.add(B);

    mean.delete();
    diff.delete();
    variance.delete();
    varEps.delete();
    varEpsSqrt.delete();
    xmean.delete();
    normalized.delete();
    scaled.delete();

    // TODO: Replace this with native batchnorm implementation

    return [result];
  }

  async toCPU() {
    this.epsTensor = await toCPU(this.epsTensor);
  }
  async toWASM() {
    this.epsTensor = await toWASM(this.epsTensor);
  }
  async toGPU() {
    this.epsTensor = await toGPU(this.epsTensor);
  }
}