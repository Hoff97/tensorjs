import {Dispatcher} from '../../../../ops/gpu/dispatcher';
import {Input, Operation} from '../../../../ops/gpu/operation';
import {poolResultShape} from '../../../../ops/util/pool';
import {defaultAllocator} from '../../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../../tensor/gpu/memory';
import {gpuConstructor, GPUTensor} from '../../../../tensor/gpu/tensor';
import {computeStrides, getSize} from '../../../../util/shape';

export interface MaxBackInfo {
  shapeGrad?: readonly number[];
  widthGrad?: number;
  heightGrad?: number;
  shapeArgMax?: readonly number[];
  widthArgMax?: number;
  heightArgMax?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  axes?: readonly number[];
  valueShape?: number[];

  poolStrides?: number[];
  poolStridesOnAxes?: number[];
  nAxes?: number;
}

export interface MaxBackInput {
  Grad: GPUTensorI;
  ArgMax: GPUTensorI;
  axes: number[];
  valueShape: number[];
}

export class MaxBackOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  MaxBackInfo,
  MaxBackInput
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
    ${this.getVarModifier('poolStrides')} int poolStrides[${this.maxRank}];
    ${this.getVarModifier('poolStridesOnAxes')} int poolStridesOnAxes[${
      this.maxRank
    }];
    ${this.getVarModifier('nAxes')} int nAxes;
    `;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: MaxBackInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int sumPos = 0;
      int gradPos = 0;
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (index[i] == -1) {
          break;
        }
        sumPos += index[i]*poolStrides[i];
        gradPos += stridesArgMax[i];
      }

      int maxIxPos = 0;
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (i >= nAxes) {
          break;
        }
        float ixAx = getValueAtPos(gradPos+i, widthArgMax, heightArgMax, ArgMax);
        maxIxPos += int(ixAx)*poolStridesOnAxes[i];
      }

      if (maxIxPos == sumPos) {
        return _Grad(index);
      } else {
        return 0.0;
      }
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['Grad', 'ArgMax'];
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'poolStrides', length: this.maxRank},
      {name: 'poolStridesOnAxes', length: this.maxRank},
      {name: 'nAxes'},
    ];
  }

  calc(input: MaxBackInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(
        this.outputShape,
        {Grad: input.Grad, ArgMax: input.ArgMax},
        undefined
      );
    }

    const poolStrides = computeStrides(input.valueShape);
    const poolStridesOnAxes = new Array(input.axes.length);
    for (let i = 0; i < input.axes.length; i++) {
      poolStridesOnAxes[i] = poolStrides[input.axes[i]];
    }
    for (let i = 0; i < poolStrides.length; i++) {
      if (!input.axes.includes(i)) {
        poolStrides[i] = 0;
      }
    }

    return this.compute(
      input.valueShape,
      {Grad: input.Grad, ArgMax: input.ArgMax},
      {
        poolStrides: this.pad(poolStrides),
        poolStridesOnAxes: this.pad(poolStridesOnAxes),
        nAxes: input.axes.length,
      }
    );
  }

  getOutputShape(input: MaxBackInput): readonly number[] {
    return input.valueShape;
  }

  compile(info: MaxBackInfo) {
    if (info.axes !== undefined && info.valueShape !== undefined) {
      const poolStrides = computeStrides(info.valueShape);
      const poolStridesOnAxes = new Array(info.axes.length);
      for (let i = 0; i < info.axes.length; i++) {
        poolStridesOnAxes[i] = poolStrides[info.axes[i]];
      }
      for (let i = 0; i < poolStrides.length; i++) {
        if (!info.axes.includes(i)) {
          poolStrides[i] = 0;
        }
      }

      info.poolStrides = poolStrides;
      info.poolStridesOnAxes = poolStridesOnAxes;
      info.nAxes = info.axes.length;

      delete info['axes'];
      delete info['valueShape'];
    }

    super.compile(info);
  }

  getCompilationInfo(input: MaxBackInput): MaxBackInfo {
    const outputShape = this.getOutputShape(input);

    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    const poolStrides = computeStrides(input.valueShape);
    const poolStridesOnAxes = new Array(input.axes.length);
    for (let i = 0; i < input.axes.length; i++) {
      poolStridesOnAxes[i] = poolStrides[input.axes[i]];
    }
    for (let i = 0; i < poolStrides.length; i++) {
      if (!input.axes.includes(i)) {
        poolStrides[i] = 0;
      }
    }

    return {
      shapeGrad: input.Grad.shape,
      widthGrad: input.Grad.memory.width,
      heightGrad: input.Grad.memory.height,

      shapeArgMax: input.ArgMax.shape,
      widthArgMax: input.ArgMax.memory.width,
      heightArgMax: input.ArgMax.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      poolStrides,
      poolStridesOnAxes,
      nAxes: input.axes.length,
    };
  }

  getInputInfoString(input: MaxBackInput): string {
    return `${input.Grad.shape}-${input.ArgMax.shape}-${input.axes}-${input.valueShape}`;
  }
}

const defaultMaxBackD = new Dispatcher(
  (dtype: DTypeGpu) => new MaxBackOperation(gpuConstructor, dtype)
);

export function maxBackGPU<DTpe extends DTypeGpu>(
  value: GPUTensor<DTpe>,
  gradient: GPUTensor<DTpe>,
  axes: number[]
) {
  let argMax = value.argMax(axes, false);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [resultShape, _] = poolResultShape(value.shape, axes, true);
  gradient = gradient.reshape(resultShape, false) as GPUTensor<DTpe>;
  argMax = argMax.reshape([...resultShape, axes.length], false);

  return defaultMaxBackD.calc(
    {
      Grad: gradient,
      ArgMax: argMax,
      axes,
      valueShape: value.shape,
    },
    value.dtype
  ) as GPUTensor<DTpe>;
}
