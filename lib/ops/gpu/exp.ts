import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Operation } from "./operation";


export interface ExpInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface ExpInput {
  input: GPUTensorI;
}

export class ExpOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, ExpInfo, ExpInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: ExpInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = exp(texture2D(X, uv));
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: ExpInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input})
  }

  compile(info: ExpInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
