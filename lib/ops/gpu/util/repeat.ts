import { defaultAllocator } from "../../../tensor/gpu/gl";
import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { Precision } from "../../../types";
import { getSize } from "../../../util/shape";
import { Input, Operation } from "../operation";


export interface RepeatInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;

  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;

  repeats?: readonly number[];
}

export interface RepeatInput {
  A: GPUTensorI;
  repeats: readonly number[];
}

export class RepeatOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, RepeatInfo, RepeatInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('repeats')} int repeats[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "repeats", length: this.maxRank }
    ];
  }

  getFragmentShader(info: RepeatInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int inIndex[${this.maxRank}];
      ${this.initIndex('inIndex')}
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (repeats[i] == -1) {
          break;
        }
        int d = index[i] / shapeA[i];
        inIndex[i] = index[i] - d*shapeA[i];
      }

      return _A(inIndex);
    }

    ${this.getDefaultMain()}
    `;
  }

  getOutputShape(input: RepeatInput): readonly number[] {
    const rank = input.A.shape.length;

    const outputShape = new Array(rank);
    for (let i = 0; i < rank; i++) {
      outputShape[i] = input.A.shape[i] * input.repeats[i];
    }

    return outputShape;
  }

  getTextureNames(): string[] {
    return ["A"];
  }

  calc(input: RepeatInput): GPUTensor {
    const outputShape = this.getOutputShape(input);

    return this.compute(outputShape, {A: input.A}, {repeats: this.copyPad(input.repeats)});
  }

  compile(info: RepeatInfo, precision: Precision) {
    if (info.shapeA !== undefined) {
      this.maxRank = info.shapeA.length;
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: RepeatInput, precision: Precision): RepeatInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(getSize(outputShape), precision);

    return {
      shapeA: input.A.shape,
      widthA: input.A.memory.width,
      heightA: input.A.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      repeats: input.repeats
    };
  }
}
