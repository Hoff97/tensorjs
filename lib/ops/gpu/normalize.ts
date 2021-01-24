import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Input, Operation } from "./operation";


export interface NormalizeOpInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;

  shapeMean?: readonly number[];
  widthMean?: number;
  heightMean?: number;

  shapeVariance?: readonly number[];
  widthVariance?: number;
  heightVariance?: number;

  shapeScale?: readonly number[];
  widthScale?: number;
  heightScale?: number;

  shapeBias?: readonly number[];
  widthBias?: number;
  heightBias?: number;

  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;

  epsilon: number;
}

export interface NormalizeOpInput {
  X: GPUTensorI;
  Mean: GPUTensorI;
  Variance: GPUTensorI;
  epsilon: number;
  Scale: GPUTensorI;
  Bias: GPUTensorI;
}

export class NormalizeOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, NormalizeOpInfo, NormalizeOpInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('epsilon')} float epsilon;
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "epsilon", type: "float" }
    ];
  }

  getFragmentShader(info: NormalizeOpInfo): string {
    return `
    float process(int[${this.maxRank}] index) {
      float result = _X(index) - _Mean(index);
      result = result / sqrt(_Variance(index) + epsilon);
      result = result * _Scale(index) + _Bias(index);
      return result;
    }

    ${this.getDefaultMain()}
    `;
  }

  getOutputShape(input: NormalizeOpInput): readonly number[] {
    return input.X.shape;
  }

  getTextureNames(): string[] {
    return ["X", "Mean", "Variance", "Scale", "Bias"];
  }

  calc(input: NormalizeOpInput): GPUTensor {
    return this.compute(input.X.shape, {
      X: input.X,
      Mean: input.Mean,
      Variance: input.Variance,
      Scale: input.Scale,
      Bias: input.Bias
    }, { epsilon: input.epsilon });
  }

  compile(info: NormalizeOpInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
