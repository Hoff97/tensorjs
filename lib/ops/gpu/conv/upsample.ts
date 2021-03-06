import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {Input, Operation} from '../operation';

export interface UpsampleInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  scales?: readonly number[];
}

export interface UpsampleInput {
  X: GPUTensorI;
  scales: readonly number[];
}

export class UpsampleOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  UpsampleInfo,
  UpsampleInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
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
    return ['X'];
  }

  getVariables() {
    return `
    ${this.getVarModifier('scales')} float scales[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [{name: 'scales', length: this.maxRank, type: 'float'}];
  }

  calc(input: UpsampleInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {X: input.X});
    }

    const resultShape = this.getOutputShape(input);

    return this.compute(
      resultShape,
      {X: input.X},
      {
        scales: this.copyPad(input.scales),
      }
    );
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

  getCompilationInfo(input: UpsampleInput): UpsampleInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    return {
      shapeX: input.X.shape,
      widthX: input.X.memory.width,
      heightX: input.X.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      scales: input.scales,
    };
  }

  getInputInfoString(input: UpsampleInput): string {
    return `${input.X.shape}-${input.scales}`;
  }
}
