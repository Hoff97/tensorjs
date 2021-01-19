import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Operation } from "./operation";


export interface CopyInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface CopyInput {
  input: GPUTensorI;
  outputShape?: readonly number[];
}

export class CopyOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, CopyInfo, CopyInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: CopyInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = texture2D(X, uv);
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: CopyInput): GPUTensor {
    let shape = input.outputShape;
    if (shape === undefined) {
      shape = input.input.shape;
    }

    return this.compute(shape, {X: input.input})
  }

  compile(info: CopyInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
