import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Operation } from "./operation";


export interface LogInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface LogInput {
  input: GPUTensorI;
}

export class LogOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, LogInfo, LogInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: LogInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = log(texture2D(X, uv));
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: LogInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input})
  }

  compile(info: LogInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
