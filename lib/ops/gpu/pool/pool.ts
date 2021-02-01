import {defaultAllocator} from '../../../tensor/gpu/gl';
import {GPUTensorConstructor, GPUTensorI} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Precision} from '../../../types';
import {computeStrides, getSize} from '../../../util/shape';
import {poolResultShape} from '../../util/pool';
import {Input, Operation} from '../operation';

export interface PoolInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  axes?: readonly number[];
  keepDims?: boolean;

  mappedInputStrides?: number[];
  sumDims?: number[];
  sumSize?: number;
}

export interface PoolInput {
  X: GPUTensorI;
  axes: number[];
  keepDims: boolean;
}

export abstract class PoolOperation<
  GPUTensor extends GPUTensorI
> extends Operation<GPUTensor, PoolInfo, PoolInput> {
  protected maxIterations = 1000000;

  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('mappedInputStrides')} int mappedInputStrides[${
      this.maxRank
    }];
    ${this.getVarModifier('mappedInputStrides')} int sumDims[${this.maxRank}];
    ${this.getVarModifier('mappedInputStrides')} int sumSize;
    `;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: PoolInfo): string {
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

      ${this.posToIndex('stridesX', 'inputIx', 'inputPos')}

      float res = 0.0;

      for (int i = 0; i < ${this.maxIterations}; i++) {
        if (i >= sumSize) {
          break;
        }
        float curr = _X(inputIx);
        if (i == 0) {
          res = ${this.init('curr')};
        } else {
          res = ${this.update('curr', 'res')};
        }

        ${this.incrementConditional('inputIx', 'shapeX', 'sumDims')}
      }

      ${this.post('res')}

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  abstract update(a: string, b: string): string;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  post(res: string) {
    return '';
  }
  init(res: string) {
    return res;
  }

  getTextureNames(): string[] {
    return ['X'];
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'mappedInputStrides', length: this.maxRank},
      {name: 'sumDims', length: this.maxRank},
      {name: 'sumSize'},
    ];
  }

  calc(input: PoolInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {X: input.X});
    }

    const [outputShape, ixMap] = poolResultShape(
      input.X.shape,
      input.axes,
      input.keepDims
    );

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
      }
    );
  }

  getOutputShape(input: PoolInput): readonly number[] {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [outputShape, ixMap] = poolResultShape(
      input.X.shape,
      input.axes,
      input.keepDims
    );
    return outputShape;
  }

  compile(info: PoolInfo, precision: Precision) {
    if (
      info.shapeX !== undefined &&
      info.axes !== undefined &&
      info.keepDims !== undefined
    ) {
      const [outputShape, ixMap] = poolResultShape(
        info.shapeX,
        info.axes,
        info.keepDims
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
      info.shapeOutput = outputShape;
      info.mappedInputStrides = mappedInputStrides;
      info.sumSize = sumSize;

      delete info['keepDims'];
      delete info['axes'];

      this.maxRank = info.shapeX.length;
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: PoolInput, precision: Precision): PoolInfo {
    const [outputShape, ixMap] = poolResultShape(
      input.X.shape,
      input.axes,
      input.keepDims
    );

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
      precision
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
    };
  }

  getInputInfoString(input: PoolInput): string {
    return `${input.X.shape}-${input.axes}-${input.keepDims}`;
  }
}
