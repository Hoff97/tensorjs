import { defaultAllocator } from "../../../tensor/gpu/gl";
import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { Precision } from "../../../types";
import { getSize } from "../../../util/shape";
import { Operation } from "../operation";


export interface ExpandInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface ExpandInput {
  input: GPUTensorI;
  outputShape: readonly number[];
}

export class ExpandOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, ExpandInfo, ExpandInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: ExpandInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      return _X(index);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: ExpandInput): GPUTensor {
    return this.compute(input.outputShape, {X: input.input})
  }

  getOutputShape(input: ExpandInput): readonly number[] {
    return input.outputShape;
  }

  compile(info: ExpandInfo, precision: Precision) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: ExpandInput, precision: Precision): ExpandInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(getSize(outputShape), precision);

    return {
      shapeX: input.input.shape,
      widthX: input.input.memory.width,
      heightX: input.input.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height
    };
  }
}
