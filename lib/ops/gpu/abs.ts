import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Precision } from "../../types";
import { Operation } from "./operation";


export interface AbsInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface AbsInput {
  input: GPUTensorI;
}

export class AbsOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, AbsInfo, AbsInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: AbsInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = abs(texture2D(X, uv));
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: AbsInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input})
  }

  compile(info: AbsInfo, precision: Precision) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info, precision);
  }

  getOutputShape(input: AbsInput): readonly number[] {
    return input.input.shape;
  }
}
