import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {Input, Operation} from '../operation';

export interface SliceInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  starts?: readonly number[];
  ends?: readonly number[];
  axes?: readonly number[];
  steps?: readonly number[];

  offsets?: number[];
}

export interface SliceInput {
  X: GPUTensorI;
  starts: number[];
  ends: number[];
  axes: number[];
  steps: number[];
}

export class SliceOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  SliceInfo,
  SliceInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: SliceInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int inIx[${this.maxRank}];
      ${this.initIndex('inIx')}
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (index[i] == -1) {
          break;
        }

        inIx[i] = index[i]*steps[i] + offsets[i];
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
    ${this.getVarModifier('offsets')} int offsets[${this.maxRank}];
    ${this.getVarModifier('steps')} int steps[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'offsets', length: this.maxRank},
      {name: 'steps', length: this.maxRank},
    ];
  }

  calc(input: SliceInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {X: input.X});
    }

    const rank = input.X.shape.length;

    const resultShape = [...input.X.shape];
    const offsets: number[] = new Array(rank).fill(0);
    const steps: number[] = new Array(rank).fill(1);
    let axIx = 0;
    for (let i = 0; i < rank && axIx < input.axes.length; i++) {
      if (i === input.axes[axIx]) {
        resultShape[i] = Math.ceil(
          (input.ends[axIx] - input.starts[axIx]) / input.steps[axIx]
        );
        offsets[i] = input.starts[axIx];
        steps[i] = input.steps[axIx];
        axIx++;
      }
    }

    return this.compute(
      resultShape,
      {X: input.X},
      {
        offsets: this.pad(offsets),
        steps: this.pad(steps),
      }
    );
  }

  getOutputShape(input: SliceInput): readonly number[] {
    const rank = input.X.shape.length;

    const resultShape = [...input.X.shape];
    let axIx = 0;
    for (let i = 0; i < rank && axIx < input.axes.length; i++) {
      if (i === input.axes[axIx]) {
        resultShape[i] = Math.ceil(
          (input.ends[axIx] - input.starts[axIx]) / input.steps[axIx]
        );
        axIx++;
      }
    }

    return resultShape;
  }

  compile(info: SliceInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;

      if (
        info.axes !== undefined &&
        info.starts !== undefined &&
        info.ends !== undefined &&
        info.steps !== undefined
      ) {
        const rank = info.shapeX.length;

        const offsets: number[] = new Array(rank).fill(0);
        const steps: number[] = new Array(rank).fill(1);
        let axIx = 0;
        for (let i = 0; i < rank && axIx < info.axes.length; i++) {
          if (i === info.axes[axIx]) {
            offsets[i] = info.starts[axIx];
            steps[i] = info.steps[axIx];
            axIx++;
          }
        }

        info.offsets = offsets;
        info.steps = steps;

        delete info['starts'];
        delete info['ends'];
        delete info['axes'];
      }
    }

    super.compile(info);
  }

  getCompilationInfo(input: SliceInput): SliceInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    const rank = input.X.shape.length;

    const resultShape = [...input.X.shape];
    const offsets: number[] = new Array(rank).fill(0);
    const steps: number[] = new Array(rank).fill(1);
    let axIx = 0;
    for (let i = 0; i < rank && axIx < input.axes.length; i++) {
      if (i === input.axes[axIx]) {
        resultShape[i] = Math.ceil(
          (input.ends[axIx] - input.starts[axIx]) / input.steps[axIx]
        );
        offsets[i] = input.starts[axIx];
        steps[i] = input.steps[axIx];
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

      offsets,
      steps,
    };
  }

  getInputInfoString(input: SliceInput): string {
    return `${input.X.shape}-${input.axes}-${input.starts}-${input.ends}-${input.steps}`;
  }
}
