import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { PadMode } from "../../types";
import { Input, Operation } from "./operation";


export interface UpsampleInfo {
  shapeX?: number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: number[],
  widthOutput?: number;
  heightOutput?: number;

  scales?: number[];
}

export interface UpsampleInput {
  X: GPUTensorI;
  scales: number[];
}

export class UpsampleOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, UpsampleInfo, UpsampleInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: UpsampleInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int inIx[${this.maxRank}];
      ${this.initIndex('inIx')}

      for (int i = 0; i < ${this.maxRank}; i++) {
        if (index[i] == -1) {
          break;
        }

        inIx[i] = int(floor(float(index[i]) / scales[i]));
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
    ${this.getVarModifier('scales')} float scales[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "scales", length: this.maxRank, type: "float" }
    ];
  }

  calc(input: UpsampleInput): GPUTensor {
    const resultShape = this.getOutputShape(input);

    return this.compute(resultShape, {X: input.X}, {
      scales: this.copyPad(input.scales)
    });
  }

  getOutputShape(input: UpsampleInput): readonly number[] {
    const rank = input.X.shape.length;

    const resultShape = [...input.X.shape];
    for (let i = 0; i < rank; i++) {
      resultShape[i] = Math.floor(resultShape[i] * input.scales[i]);
    }

    return resultShape;
  }

  compile(info: UpsampleInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
