import { defaultAllocator } from "../../../tensor/gpu/gl";
import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { PadMode, Precision } from "../../../types";
import { getSize } from "../../../util/shape";
import { Input, Operation } from "../operation";


export interface PadInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;

  pads?: number[];
  mode?: PadMode | number;
  value?: number;
}

export interface PadInput {
  input: GPUTensorI;
  pads: number[];
  mode: PadMode;
  value: number;
}

export class PadOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, PadInfo, PadInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: PadInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int inputIx[${this.maxRank}];
      ${this.initIndex('inputIx')}
      if (mode == 0) {
        float res = value;

        int outOfBounds = 0;
        for (int i = 0; i < ${this.maxRank}; i++) {
          if (index[i] == -1) {
            break;
          }
          inputIx[i] = index[i] - pads[i];
          if (inputIx[i] < 0 || inputIx[i] >= shapeX[i]) {
            outOfBounds = 1;
            break;
          }
        }

        if (outOfBounds == 0) {
          res = _X(inputIx);
        }

        return res;
      } else if (mode == 1) {
        for (int i = 0; i < ${this.maxRank}; i++) {
          if (index[i] == -1) {
            break;
          }
          inputIx[i] = index[i] - pads[i];
          if (inputIx[i] < 0) {
            inputIx[i] = -inputIx[i];
          } else if (inputIx[i] >= shapeX[i]) {
            inputIx[i] = 2*shapeX[i] - inputIx[i] - 2;
          }
        }

        return _X(inputIx);
      } else {
        for (int i = 0; i < ${this.maxRank}; i++) {
          if (index[i] == -1) {
            break;
          }
          inputIx[i] = index[i] - pads[i];
          if (inputIx[i] < 0) {
            inputIx[i] = 0;
          } else if (inputIx[i] >= shapeX[i]) {
            inputIx[i] = shapeX[i] - 1;
          }
        }

        return _X(inputIx);
      }
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  getVariables() {
    return `
    ${this.getVarModifier('pads')} int pads[${this.maxRank*2}];
    ${this.getVarModifier('value')} float value;
    ${this.getVarModifier('mode')} int mode;
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "value", type: "float" },
      { name: "pads", length: this.maxRank*2 },
      { name: "mode" }
    ];
  }

  getModeFlag(mode: PadMode) {
    return mode === "constant" ? 0 : mode === "reflect" ? 1 : 2;
  }

  calc(input: PadInput): GPUTensor {
    const resultShape = this.getOutputShape(input);

    return this.compute(resultShape, {X: input.input}, {
      pads: this.copyPad(input.pads, this.maxRank*2),
      value: input.value,
      mode: this.getModeFlag(input.mode)
    });
  }

  getOutputShape(input: PadInput): readonly number[] {
    const rank = input.input.shape.length;

    const resultShape = [...input.input.shape];
    for (let i = 0; i < rank; i++) {
      resultShape[i] += input.pads[i] + input.pads[i+rank];
    }

    return resultShape;
  }

  compile(info: PadInfo, precision: Precision) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    if (info.mode !== undefined && typeof info.mode === 'string') {
      info.mode = this.getModeFlag(info.mode as any);
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: PadInput, precision: Precision): PadInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(getSize(outputShape), precision);

    return {
      shapeX: input.input.shape,
      widthX: input.input.memory.width,
      heightX: input.input.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      pads: input.pads,
      mode: this.getModeFlag(input.mode),
      value: input.value
    };
  }
}
