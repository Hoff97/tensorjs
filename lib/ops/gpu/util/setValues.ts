import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {Input, Operation} from '../operation';

export interface SetValuesInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;

  shapeValues?: readonly number[];
  widthValues?: number;
  heightValues?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  starts?: number[];
}

export interface SetValuesInput {
  A: GPUTensorI;
  Values: GPUTensorI;
  starts: number[];
}

export class SetValuesOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  SetValuesInfo,
  SetValuesInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('starts')} int starts[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [{name: 'starts', length: this.maxRank}];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: SetValuesInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int valueIx[${this.maxRank}];
      ${this.initIndex('valueIx')}

      int inValues = 1;

      for (int i = 0; i < ${this.maxRank}; i++) {
        if (index[i] == -1) {
          break;
        }

        if (index[i] < starts[i] || index[i] >= (starts[i] + shapeValues[i])) {
          inValues = 0;
          break;
        } else {
          valueIx[i] = index[i] - starts[i];
        }
      }

      if (inValues == 1) {
        return _Values(valueIx);
      } else {
        return _A(index);
      }
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['A', 'Values'];
  }

  calc(input: SetValuesInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {A: input.A, Values: input.Values});
    }

    const outputShape = this.getOutputShape(input);

    return this.compute(
      outputShape,
      {A: input.A, Values: input.Values},
      {starts: this.pad(input.starts)}
    );
  }

  getOutputShape(input: SetValuesInput): readonly number[] {
    return input.A.shape;
  }

  compile(info: SetValuesInfo) {
    if (info.shapeA !== undefined) {
      this.maxRank = info.shapeA.length;
    }

    super.compile(info);
  }

  getCompilationInfo(input: SetValuesInput): SetValuesInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    return {
      shapeA: input.A.shape,
      widthA: input.A.memory.width,
      heightA: input.A.memory.height,

      shapeValues: input.Values.shape,
      widthValues: input.Values.memory.width,
      heightValues: input.Values.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      starts: input.starts,
    };
  }

  getInputInfoString(input: SetValuesInput): string {
    return `${input.A.shape}-${input.Values.shape}-${input.starts}`;
  }
}
