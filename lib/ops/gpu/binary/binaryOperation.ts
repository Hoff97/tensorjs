import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { Precision } from "../../../types";
import { Operation } from "./../operation";


export interface BinaryOpInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;
  shapeB?: readonly number[];
  widthB?: number;
  heightB?: number;
  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface BinaryOpInput {
  A: GPUTensorI;
  B: GPUTensorI;
  outputShape: readonly number[];
}

export abstract class BinaryOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, BinaryOpInfo, BinaryOpInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  abstract getOp(a: string, b: string): string;

  getFragmentShader(info: BinaryOpInfo): string {
    return `
    float process(int[${this.maxRank}] index) {
      return ${this.getOp('_A(index)', '_B(index)')};
    }

    ${this.getDefaultMain()}
    `;
  }

  getOutputShape(input: BinaryOpInput): readonly number[] {
    return input.outputShape;
  }

  getTextureNames(): string[] {
    return ["A", "B"];
  }

  calc(input: BinaryOpInput): GPUTensor {
    return this.compute(input.outputShape, {A: input.A, B: input.B})
  }

  compile(info: BinaryOpInfo, precision: Precision) {
    if (info.shapeA !== undefined) {
      this.maxRank = info.shapeA.length;
    }
    if (info.shapeB !== undefined) {
      this.maxRank = info.shapeB.length;
    }

    super.compile(info, precision);
  }
}
