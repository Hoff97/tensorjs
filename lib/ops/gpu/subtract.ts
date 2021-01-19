import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Operation } from "./operation";


export interface SubtractInfo {
  shapeA?: number[];
  widthA?: number;
  heightA?: number;
  shapeB?: number[];
  widthB?: number;
  heightB?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface SubtractInput {
  A: GPUTensorI;
  B: GPUTensorI;
  outputShape: readonly number[];
}

export class SubtractOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, SubtractInfo, SubtractInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: SubtractInfo): string {
    return `
    float process(int[${this.maxRank}] index) {
      return _A(index) - _B(index);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["A", "B"];
  }

  calc(input: SubtractInput): GPUTensor {
    return this.compute(input.outputShape, {A: input.A, B: input.B})
  }

  compile(info: SubtractInfo) {
    if (info.shapeA !== undefined) {
      this.maxRank = info.shapeA.length;
    }
    if (info.shapeB !== undefined) {
      this.maxRank = info.shapeB.length;
    }

    super.compile(info);
  }
}
