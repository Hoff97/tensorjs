import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Operation } from "./operation";


export interface ExpandInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface ExpandInput {
  input: GPUTensorI;
  outputShape: readonly number[];
}

export class ExpandOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, ExpandInfo, ExpandInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: ExpandInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      return _X(index);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: ExpandInput): GPUTensor {
    return this.compute(input.outputShape, {X: input.input})
  }

  getOutputShape(input: ExpandInput): readonly number[] {
    return input.outputShape;
  }

  compile(info: ExpandInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
