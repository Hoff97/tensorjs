import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Operation } from "./operation";


export interface SqrtInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface SqrtInput {
  input: GPUTensorI;
}

export class SqrtOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, SqrtInfo, SqrtInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: SqrtInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = sqrt(texture2D(X, uv));
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: SqrtInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input})
  }

  compile(info: SqrtInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
