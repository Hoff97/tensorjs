import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { Precision } from "../../../types";
import { Operation } from "../operation";


export interface UnaryOpInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface UnaryOpInput {
  input: GPUTensorI;
}

export abstract class UnaryOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, UnaryOpInfo, UnaryOpInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  abstract operation(input: string): string;

  getFragmentShader(info: UnaryOpInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = ${this.operation('texture2D(X, uv)')};
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: UnaryOpInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input})
  }

  getOutputShape(input: UnaryOpInput): readonly number[] {
    return input.input.shape;
  }

  compile(info: UnaryOpInfo, precision: Precision) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info, precision);
  }
}
