import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { PadMode } from "../../types";
import { Input, Operation } from "./operation";


export interface SliceInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface SliceInput {
  X: GPUTensorI;
  starts: number[];
  ends: number[];
  axes: number[];
}

export class SliceOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, SliceInfo, SliceInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: SliceInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int inIx[${this.maxRank}];
      ${this.initIndex('inIx')}
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (index[i] == -1) {
          break;
        }

        inIx[i] = index[i] + offsets[i];
      }

      return _X(inIx);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  getVariables() {
    return `
    ${this.getVarModifier('offsets')} int offsets[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "offsets", length: this.maxRank*2 }
    ];
  }

  calc(input: SliceInput): GPUTensor {
    const rank = input.X.shape.length;

    const resultShape = [...input.X.shape];
    const offsets: number[] = new Array(rank).fill(0);
    let axIx = 0;
    for (let i = 0; i < rank && axIx < input.axes.length; i++) {
      if (i == input.axes[axIx]) {
        resultShape[i] = input.ends[axIx] - input.starts[axIx];
        offsets[i] = input.starts[axIx];
        axIx++;
      }
    }

    return this.compute(resultShape, {X: input.X}, {
      offsets: this.pad(offsets)
    });
  }

  compile(info: SliceInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
