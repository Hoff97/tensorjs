import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Operation } from "./operation";


export interface CeilInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface CeilInput {
  input: GPUTensorI;
}

export class CeilOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, CeilInfo, CeilInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: CeilInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = ceil(texture2D(X, uv));
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: CeilInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input})
  }

  compile(info: CeilInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
