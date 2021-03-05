import {getSize} from '../../../util/shape';

import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {defaultAllocator} from '../../../tensor/gpu/gl';
import {Input, Operation} from '../../../ops/gpu/operation';
import {Dispatcher} from '../../../ops/gpu/dispatcher';
import {gpuConstructor} from '../../../tensor/gpu/tensor';

export interface UpdateMomentInfo {
  shapeGrad?: readonly number[];
  widthGrad?: number;
  heightGrad?: number;
  shapeMoments?: readonly number[];
  widthMoments?: number;
  heightMoments?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  beta1?: number;
  beta2?: number;
}

export interface UpdateMomentInput {
  Grad: GPUTensorI;
  Moments: GPUTensorI;
  beta1: number;
  beta2: number;
  t: number;
}

export class UpdateMomentOperation<
  GPUTensor extends GPUTensorI
> extends Operation<GPUTensor, UpdateMomentInfo, UpdateMomentInput> {
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
    ${this.getVarModifier('beta1')} float beta1;
    ${this.getVarModifier('beta2')} float beta2;
    ${this.getVarModifier('t')} int t;
    `;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: UpdateMomentInfo): string {
    return `
    void main() {
      initVars();

      int pos = coordinateToPos(uv, widthOutput, heightOutput);

      float m1 = getValueAtPos(pos, widthMoments, heightMoments, Moments);
      float m2 = getValueAtPos(pos+2, widthMoments, heightMoments, Moments);
      float grad = getValueAtPos(pos/4, widthGrad, heightGrad, Grad);

      m1 = beta1*m1 + (1.0-beta1)*grad;
      m2 = beta2*m2 + (1.0-beta2)*grad*grad;

      float m1Corr = m1/(1.0-pow(beta1, float(t)));
      float m2Corr = m2/(1.0-pow(beta2, float(t)));

      gl_FragColor = vec4(m1,m1Corr,m2,m2Corr);
    }
    `;
  }

  getTextureNames(): string[] {
    return ['Grad', 'Moments'];
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'beta1', type: 'float'},
      {name: 'beta2', type: 'float'},
      {name: 't', type: 'int'},
    ];
  }

  calc(input: UpdateMomentInput): GPUTensor {
    return this.compute(
      input.Moments.shape,
      {Grad: input.Grad, Moments: input.Moments},
      {
        beta1: input.beta1,
        beta2: input.beta2,
        t: input.t,
      }
    );
  }

  getOutputShape(input: UpdateMomentInput): readonly number[] {
    return input.Moments.shape;
  }

  compile(info: UpdateMomentInfo) {
    if (info.shapeMoments !== undefined) {
      this.maxRank = info.shapeMoments.length;
    }

    super.compile(info);
  }

  getCompilationInfo(input: UpdateMomentInput): UpdateMomentInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    return {
      shapeGrad: input.Grad.shape,
      widthGrad: input.Grad.memory.width,
      heightGrad: input.Grad.memory.height,

      shapeMoments: input.Moments.shape,
      widthMoments: input.Moments.memory.width,
      heightMoments: input.Moments.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      beta1: input.beta1,
      beta2: input.beta2,
    } as UpdateMomentInfo;
  }

  getInputInfoString(input: UpdateMomentInput): string {
    return `${input.Grad.shape}-${input.Moments.shape}-${input.beta1}-${input.beta2}`;
  }
}

export const defaultUpdateMomentsD = new Dispatcher(
  (dtype: DTypeGpu) => new UpdateMomentOperation(gpuConstructor, dtype)
);
