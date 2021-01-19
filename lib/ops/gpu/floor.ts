import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Operation } from "./operation";


export interface FloorInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface FloorInput {
  input: GPUTensorI;
}

export class FloorOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, FloorInfo, FloorInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: FloorInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = floor(texture2D(X, uv));
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: FloorInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input})
  }

  compile(info: FloorInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
