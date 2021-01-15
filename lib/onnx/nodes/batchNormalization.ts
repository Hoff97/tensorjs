import { CPUTensor } from "../../tensor/cpu/tensor";
import { glContext } from "../../tensor/gpu/gl";
import Tensor from "../../types";
import { toCPU, toGPU, toWASM } from "../../util/convert";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class BatchNormalizationNode extends OnnxNode {
  private epsilon: number;
  private momentum: number;

  private epsTensor: Tensor;

  private scale?: Tensor;
  private bias?: Tensor;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.epsilon = this.getAttributeFloat('epsilon') || 1e-05;
    this.momentum = this.getAttributeFloat('momentum') || 0.9;

    this.epsTensor = new CPUTensor([1], [this.epsilon]);

    //TODO: Handle lower onnxversions here
  }

  initialize(resolveConstant: (name: string) => Tensor) {
    const scale = resolveConstant(this.inputs[1]);
    const B = resolveConstant(this.inputs[2]);
    const mean = resolveConstant(this.inputs[3]);
    const variance = resolveConstant(this.inputs[4]);

    if (scale !== undefined && B !== undefined && mean !== undefined && variance !== undefined) {
      const varSqrt = variance.add(this.epsTensor).sqrt();

      this.scale = scale.divide(varSqrt);
      this.bias = B.subtract(mean.multiply(this.scale));

      varSqrt.delete();
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (this.scale !== undefined) {
      const scaled = x.multiply(this.scale);
      const result = scaled.add(this.bias);
      scaled.delete();

      return [result];
    } else {
      let scale = inputs[1];
      let B = inputs[2];
      let mean = inputs[3];
      let variance = inputs[4];

      //TODO: Handle lower onnx versions here

      const C = scale.getShape()[0];

      const newShape = [1,C,...new Array(x.getShape().length - 2).fill(1)];

      scale = scale.reshape(newShape, false);
      B = B.reshape(newShape, false);
      mean = mean.reshape(newShape, false);
      variance = variance.reshape(newShape, false);

      const result = x.normalize(mean, variance, this.epsilon, scale, B);

      return [result];
    }
  }

  async toCPU() {
    if (this.scale !== undefined) {
      this.scale = await toCPU(this.scale);
      this.bias = await toCPU(this.bias);
    }
  }

  async toWASM() {
    if (this.scale !== undefined) {
      this.scale = await toWASM(this.scale);
      this.bias = await toWASM(this.bias);
    }
  }

  async toGPU() {
    if (this.scale !== undefined) {
      this.scale = await toGPU(this.scale);
      this.bias = await toGPU(this.bias);
    }
  }
}