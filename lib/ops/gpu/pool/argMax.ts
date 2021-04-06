import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {computeStrides, getSize} from '../../../util/shape';
import {poolResultShape} from '../../util/pool';
import {Input, Operation} from '../operation';

export interface ArgMaxInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  axes?: readonly number[];
  selectLast?: number;

  mappedInputStrides?: number[];
  sumDims?: number[];
  sumSize?: number;
}

export interface ArgMaxInput {
  X: GPUTensorI;
  axes: number[];
  selectLast: boolean;
}

export class ArgMaxOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  ArgMaxInfo,
  ArgMaxInput
> {
  protected maxIterations = 1000000;

  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('mappedInputStrides')} int mappedInputStrides[${
      this.maxRank
    }];
    ${this.getVarModifier('sumDims')} int sumDims[${this.maxRank}];
    ${this.getVarModifier('axes')} int axes[${this.maxRank}];
    ${this.getVarModifier('sumSize')} int sumSize;
    ${this.getVarModifier('selectLast')} int selectLast;
    `;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: ArgMaxInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int inputIx[${this.maxRank}];
      ${this.initIndex('inputIx')}

      int inputPos = 0;
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (mappedInputStrides[i] == -1 || index[i] == -1) {
          break;
        }
        inputPos += mappedInputStrides[i]*index[i];
      }

      int axResult = -1;
      for (int j = 0; j < ${this.maxRank}; j++) {
        if (j == rankOutput - 1) {
          axResult = index[j];
          break;
        }
      }
      for (int j = 0; j < ${this.maxRank}; j++) {
        if (j == axResult) {
          axResult = axes[j];
          break;
        }
      }

      ${this.posToIndex('stridesX', 'inputIx', 'inputPos')}

      float res = 0.0;
      int ixResult = -1;

      for (int i = 0; i < ${this.maxIterations}; i++) {
        if (i >= sumSize) {
          break;
        }
        float curr = _X(inputIx);
        if (i == 0) {
          res = curr;
          for (int j = 0; j < ${this.maxRank}; j++) {
            if (j == axResult) {
              ixResult = inputIx[j];
              break;
            }
          }
        } else {
          if (curr > res || (res == curr && selectLast == 1)) {
            res = curr;
            for (int j = 0; j < ${this.maxRank}; j++) {
              if (j == axResult) {
                ixResult = inputIx[j];
                break;
              }
            }
          }
        }

        ${this.incrementConditional('inputIx', 'shapeX', 'sumDims')}
      }

      return float(ixResult);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['X'];
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'mappedInputStrides', length: this.maxRank},
      {name: 'sumDims', length: this.maxRank},
      {name: 'axes', length: this.maxRank},
      {name: 'sumSize'},
      {name: 'selectLast'},
    ];
  }

  calc(input: ArgMaxInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {X: input.X}, undefined, 'uint32');
    }

    // eslint-disable-next-line prefer-const
    let [outputShape, ixMap] = poolResultShape(
      input.X.shape,
      input.axes,
      false
    );

    outputShape = [...outputShape, input.axes.length];

    const inputStrides = computeStrides(input.X.shape);
    const mappedInputStrides = [];
    for (const i of ixMap) {
      mappedInputStrides.push(inputStrides[i]);
    }

    let sumSize = 1;
    const sumDims: number[] = new Array(input.X.shape.length).fill(0);
    for (let i = 0; i < input.axes.length; i++) {
      sumDims[input.axes[i]] = 1;
      sumSize *= input.X.shape[input.axes[i]];
    }

    return this.compute(
      outputShape,
      {X: input.X},
      {
        mappedInputStrides: this.pad(mappedInputStrides),
        sumDims: this.pad(sumDims),
        sumSize,
        selectLast: input.selectLast ? 1 : 0,
        axes: this.copyPad(input.axes),
      },
      'uint32'
    );
  }

  getOutputShape(input: ArgMaxInput): readonly number[] {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [outputShape, ixMap] = poolResultShape(
      input.X.shape,
      input.axes,
      false
    );

    return [...outputShape, input.axes.length];
  }

  compile(info: ArgMaxInfo) {
    if (
      info.shapeX !== undefined &&
      info.axes !== undefined &&
      info.selectLast !== undefined
    ) {
      const [outputShape, ixMap] = poolResultShape(
        info.shapeX,
        info.axes,
        false
      );

      const inputStrides = computeStrides(info.shapeX);
      const mappedInputStrides = [];
      for (const i of ixMap) {
        mappedInputStrides.push(inputStrides[i]);
      }

      let sumSize = 1;
      const sumDims: number[] = new Array(info.shapeX.length).fill(0);
      for (let i = 0; i < info.axes.length; i++) {
        sumDims[info.axes[i]] = 1;
        sumSize *= info.shapeX[info.axes[i]];
      }

      info.sumDims = sumDims;
      info.shapeOutput = [...outputShape, info.axes.length];
      info.mappedInputStrides = mappedInputStrides;
      info.sumSize = sumSize;
    }

    super.compile(info);
  }

  getCompilationInfo(input: ArgMaxInput): ArgMaxInfo {
    // eslint-disable-next-line prefer-const
    let [outputShape, ixMap] = poolResultShape(
      input.X.shape,
      input.axes,
      false
    );

    outputShape = [...outputShape, input.axes.length];

    const inputStrides = computeStrides(input.X.shape);
    const mappedInputStrides = [];
    for (const i of ixMap) {
      mappedInputStrides.push(inputStrides[i]);
    }

    let sumSize = 1;
    const sumDims: number[] = new Array(input.X.shape.length).fill(0);
    for (let i = 0; i < input.axes.length; i++) {
      sumDims[input.axes[i]] = 1;
      sumSize *= input.X.shape[input.axes[i]];
    }

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

      mappedInputStrides,
      sumDims,
      sumSize,
      selectLast: input.selectLast ? 1 : 0,
      axes: input.axes,
    };
  }

  getInputInfoString(input: ArgMaxInput): string {
    return `${input.X.shape}-${input.axes}-${input.selectLast}`;
  }
}
