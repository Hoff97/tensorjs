import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { computeStrides } from "../../util/shape";
import { Input, Operation } from "./operation";


export interface TransposeInfo {
  shapeA?: number[];
  widthA?: number;
  heightA?: number;

  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;

  permutation?: number[];
  mappedStrides?: number[];
}

export interface TransposeInput {
  A: GPUTensorI;
  permutation: number[];
}

export class TransposeOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, TransposeInfo, TransposeInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('mappedStrides')} int mappedStrides[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "mappedStrides", length: this.maxRank }
    ];
  }

  getFragmentShader(info: TransposeInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      return getValueAt(index, mappedStrides, widthA, heightA, A);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["A"];
  }

  calc(input: TransposeInput): GPUTensor {
    const rank = input.A.shape.length;

    const outputShape = new Array(rank);
    for (let i = 0; i < rank; i++) {
      outputShape[i] = input.A.shape[input.permutation[i]];
    }

    const inputStrides = computeStrides(input.A.shape);
    const mappedStrides = new Array(rank);
    for (let i = 0; i < rank; i++) {
      mappedStrides[i] = inputStrides[input.permutation[i]];
    }

    return this.compute(outputShape, {A: input.A}, {mappedStrides: this.pad(mappedStrides)});
  }

  compile(info: TransposeInfo) {
    if (info.shapeA !== undefined) {
      this.maxRank = info.shapeA.length;
    }

    super.compile(info);
  }
}
