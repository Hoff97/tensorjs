import { CPUTensor } from "../../tensor/cpu/tensor";
import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { computeStrides } from "../../util/shape";
import { Input, Operation } from "./operation";


export interface GatherInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;

  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;

  axis?: number;
  indexValues?: number[];
  mappedInputStrides?: number[];
  mappedIndexStrides?: number[];
}

export interface GatherInput {
  X: GPUTensorI;
  indices: CPUTensor;
  axis: number;
}

export class GatherOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, GatherInfo, GatherInput> {
  protected gatherMaxIxSize = 10;

  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('axis')} int axis;
    ${this.getVarModifier('indexValues')} int indexValues[${this.gatherMaxIxSize}];

    ${this.getVarModifier('mappedIndexStrides')} int mappedIndexStrides[${this.maxRank}];
    ${this.getVarModifier('mappedInputStrides')} int mappedInputStrides[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "axis" },
      { name: "indexValues", length: this.gatherMaxIxSize },
      { name: "mappedInputStrides", length: this.maxRank },
      { name: "mappedIndexStrides", length: this.maxRank }
    ];
  }

  getFragmentShader(info: GatherInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int inputPos = 0;
      int indexPos = 0;

      int strideAxis = 0;
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (index[i] == -1) {
          break;
        }
        if (i == axis) {
          strideAxis = stridesX[i];
        }
        inputPos += mappedInputStrides[i]*index[i];
        indexPos += mappedIndexStrides[i]*index[i];
      }

      for (int i = 0; i < ${this.gatherMaxIxSize}; i++) {
        if (i == indexPos) {
          inputPos += indexValues[i]*strideAxis;
          break;
        }
      }

      return getValueAtPos(inputPos, widthX, heightX, X);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: GatherInput): GPUTensor {
    if (input.indices.size > this.gatherMaxIxSize) {
      throw new Error(`Gather on GPU can deal with at most ${this.gatherMaxIxSize} indices, input had ${input.indices.size}`);
    }

    const r = input.X.shape.length;
    const q = input.indices.shape.length;

    const inputStrides = computeStrides(input.X.shape);
    const indexStrides = computeStrides(input.indices.shape);

    const resultRank = r + q - 1;
    const resultShape = new Array(resultRank);

    const mappedInputStrides = new Array(resultRank).fill(0);
    const mappedIndexStrides = new Array(resultRank).fill(0);

    for (let i = 0; i < input.axis; i++) {
      resultShape[i] = input.X.shape[i];
      mappedInputStrides[i] = inputStrides[i];

      mappedIndexStrides[i] = 0;
    }
    for (let i = 0; i < q; i++) {
      resultShape[i + input.axis] = input.indices.shape[i];
      mappedIndexStrides[i + input.axis] = indexStrides[i];

      mappedInputStrides[i + input.axis] = 0;
    }
    for (let i = input.axis + 1; i < r; i++) {
      resultShape[i + q - 1] = input.X.shape[i];
      mappedInputStrides[i + q - 1] = inputStrides[i];

      mappedIndexStrides[i + q - 1] = 0;
    }

    return this.compute(resultShape, {X: input.X}, {
      axis: input.axis,
      indexValues: this.pad(Array.from(input.indices.values), this.gatherMaxIxSize),
      mappedInputStrides: this.pad(mappedInputStrides),
      mappedIndexStrides: this.pad(mappedIndexStrides)
    })
  }

  getOutputShape(input: GatherInput): readonly number[] {
    const r = input.X.shape.length;
    const q = input.indices.shape.length;

    const resultRank = r + q - 1;
    const resultShape = new Array(resultRank);

    for (let i = 0; i < input.axis; i++) {
      resultShape[i] = input.X.shape[i];
    }
    for (let i = 0; i < q; i++) {
      resultShape[i + input.axis] = input.indices.shape[i];
    }
    for (let i = input.axis + 1; i < r; i++) {
      resultShape[i + q - 1] = input.X.shape[i];
    }

    return resultShape;
  }

  compile(info: GatherInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }
    // TODO!

    super.compile(info);
  }
}
