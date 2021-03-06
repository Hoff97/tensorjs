import {CPUTensor} from '../../../tensor/cpu/tensor';
import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {computeStrides, getSize} from '../../../util/shape';
import {Input, Operation} from '../operation';

export interface GatherInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  indices?: CPUTensor<'uint32'>;

  axis?: number;
  indexValues?: number[];
  mappedInputStrides?: number[];
  mappedIndexStrides?: number[];
}

export interface GatherInput {
  X: GPUTensorI;
  indices: CPUTensor<'uint32'>;
  axis: number;
}

export class GatherOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  GatherInfo,
  GatherInput
> {
  protected gatherMaxIxSize = 10;

  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('axis')} int axis;
    ${this.getVarModifier('indexValues')} int indexValues[${
      this.gatherMaxIxSize
    }];
    ${this.getVarModifier('mappedIndexStrides')} int mappedIndexStrides[${
      this.maxRank
    }];
    ${this.getVarModifier('mappedInputStrides')} int mappedInputStrides[${
      this.maxRank
    }];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'axis'},
      {name: 'indexValues', length: this.gatherMaxIxSize},
      {name: 'mappedInputStrides', length: this.maxRank},
      {name: 'mappedIndexStrides', length: this.maxRank},
    ];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
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
    return ['X'];
  }

  calc(input: GatherInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {X: input.X});
    }

    if (input.indices.size > this.gatherMaxIxSize) {
      throw new Error(
        `Gather on GPU can deal with at most ${this.gatherMaxIxSize} indices, input had ${input.indices.size}`
      );
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

    return this.compute(
      resultShape,
      {X: input.X},
      {
        axis: input.axis,
        indexValues: this.pad(
          Array.from(input.indices.values),
          this.gatherMaxIxSize
        ),
        mappedInputStrides: this.pad(mappedInputStrides),
        mappedIndexStrides: this.pad(mappedIndexStrides),
      }
    );
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

      if (info.indices !== undefined && info.axis !== undefined) {
        const r = info.shapeX.length;
        const q = info.indices.shape.length;

        const inputStrides = computeStrides(info.shapeX);
        const indexStrides = computeStrides(info.indices.shape);

        const resultRank = r + q - 1;
        const resultShape = new Array(resultRank);

        const mappedInputStrides = new Array(resultRank).fill(0);
        const mappedIndexStrides = new Array(resultRank).fill(0);

        for (let i = 0; i < info.axis; i++) {
          resultShape[i] = info.shapeX[i];
          mappedInputStrides[i] = inputStrides[i];

          mappedIndexStrides[i] = 0;
        }
        for (let i = 0; i < q; i++) {
          resultShape[i + info.axis] = info.indices.shape[i];
          mappedIndexStrides[i + info.axis] = indexStrides[i];

          mappedInputStrides[i + info.axis] = 0;
        }
        for (let i = info.axis + 1; i < r; i++) {
          resultShape[i + q - 1] = info.shapeX[i];
          mappedInputStrides[i + q - 1] = inputStrides[i];

          mappedIndexStrides[i + q - 1] = 0;
        }

        info.mappedIndexStrides = mappedIndexStrides;
        info.mappedInputStrides = mappedInputStrides;
        info.indexValues = Array.from(info.indices.values);

        delete info['indices'];
      }
    }

    super.compile(info);
  }

  getCompilationInfo(input: GatherInput): GatherInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

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

    return {
      shapeX: input.X.shape,
      widthX: input.X.memory.width,
      heightX: input.X.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      axis: input.axis,
      indexValues: Array.from(input.indices.values),
      mappedIndexStrides,
      mappedInputStrides,
    };
  }

  getInputInfoString(input: GatherInput): string {
    return `${input.X.shape}-${input.axis}-${Array.from(
      input.indices.values
    )}-${input.indices.shape}`;
  }
}
