import { defaultAllocator } from "../../../tensor/gpu/gl";
import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { Precision } from "../../../types";
import { getSize } from "../../../util/shape";
import { Input, Operation } from "../operation";


export interface SliceInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;

  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;

  starts?: readonly number[];
  ends?: readonly number[];
  axes?: readonly number[];

  offsets?: number[];
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
      { name: "offsets", length: this.maxRank }
    ];
  }

  calc(input: SliceInput): GPUTensor {
    if (this.fullyStatic) {
      return this.compute(this.outputShape, {X: input.X});
    }

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

  getOutputShape(input: SliceInput): readonly number[] {
    const rank = input.X.shape.length;

    const resultShape = [...input.X.shape];
    let axIx = 0;
    for (let i = 0; i < rank && axIx < input.axes.length; i++) {
      if (i == input.axes[axIx]) {
        resultShape[i] = input.ends[axIx] - input.starts[axIx];
        axIx++;
      }
    }

    return resultShape;
  }

  compile(info: SliceInfo, precision: Precision) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;

      if (info.axes !== undefined && info.starts !== undefined && info.starts !== undefined) {
        const rank = info.shapeX.length;

        const offsets: number[] = new Array(rank).fill(0);
        let axIx = 0;
        for (let i = 0; i < rank && axIx < info.axes.length; i++) {
          if (i == info.axes[axIx]) {
            offsets[i] = info.starts[axIx];
            axIx++;
          }
        }

        info.offsets = offsets;

        delete info['starts'];
        delete info['ends'];
        delete info['axes'];
      }
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: SliceInput, precision: Precision): SliceInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(getSize(outputShape), precision);

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

    return {
      shapeX: input.X.shape,
      widthX: input.X.memory.width,
      heightX: input.X.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      offsets
    };
  }

  getInputInfoString(input: SliceInput): string {
    return `${input.X.shape}-${input.axes}-${input.starts}-${input.ends}`;
  }
}
