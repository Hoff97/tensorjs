import {getSize} from '../../../util/shape';

import {GPUTensorConstructor, GPUTensorI} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Precision} from '../../../types';
import {defaultAllocator} from '../../../tensor/gpu/gl';
import {Input, Operation} from '../../../ops/gpu/operation';
import {Dispatcher} from '../../../ops/gpu/dispatcher';
import {gpuConstructor} from '../../../tensor/gpu/tensor';

export interface UpdateValueInfo {
  shapeValue?: readonly number[];
  widthValue?: number;
  heightValue?: number;

  shapeMoments?: readonly number[];
  widthMoments?: number;
  heightMoments?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  alpha?: number;
  epsilon?: number;
}

export interface UpdateValueInput {
  Value: GPUTensorI;
  Moments: GPUTensorI;
  alpha: number;
  epsilon: number;
}

export class UpdateValueOperation<
  GPUTensor extends GPUTensorI
> extends Operation<GPUTensor, UpdateValueInfo, UpdateValueInput> {
  protected maxIterations = 1000000;

  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('alpha')} float alpha;
    ${this.getVarModifier('epsilon')} float epsilon;
    `;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: UpdateValueInfo): string {
    return `
    float newVal(float m1Corr, float m2Corr, float value) {
      return value - alpha*(m1Corr/(sqrt(m2Corr)+epsilon));
    }

    void main() {
      initVars();

      int pos = coordinateToPos(uv, widthOutput, heightOutput);

      vec4 result = vec4(0,0,0,0);

      float m1Corr = getValueAtPos(pos*4+1, widthMoments, heightMoments, Moments);
      float m2Corr = getValueAtPos(pos*4+3, widthMoments, heightMoments, Moments);
      float value = getValueAtPos(pos, widthValue, heightValue, Value);
      result.r = newVal(m1Corr, m2Corr, value);

      pos++;
      m1Corr = getValueAtPos(pos*4+1, widthMoments, heightMoments, Moments);
      m2Corr = getValueAtPos(pos*4+3, widthMoments, heightMoments, Moments);
      value = getValueAtPos(pos, widthValue, heightValue, Value);
      result.g = newVal(m1Corr, m2Corr, value);

      pos++;
      m1Corr = getValueAtPos(pos*4+1, widthMoments, heightMoments, Moments);
      m2Corr = getValueAtPos(pos*4+3, widthMoments, heightMoments, Moments);
      value = getValueAtPos(pos, widthValue, heightValue, Value);
      result.b = newVal(m1Corr, m2Corr, value);

      pos++;
      m1Corr = getValueAtPos(pos*4+1, widthMoments, heightMoments, Moments);
      m2Corr = getValueAtPos(pos*4+3, widthMoments, heightMoments, Moments);
      value = getValueAtPos(pos, widthValue, heightValue, Value);
      result.a = newVal(m1Corr, m2Corr, value);


      gl_FragColor = result;
    }
    `;
  }

  getTextureNames(): string[] {
    return ['Value', 'Moments'];
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'alpha', type: 'float'},
      {name: 'epsilon', type: 'float'},
    ];
  }

  calc(input: UpdateValueInput): GPUTensor {
    return this.compute(
      input.Value.shape,
      {Value: input.Value, Moments: input.Moments},
      {
        alpha: input.alpha,
        epsilon: input.epsilon,
      }
    );
  }

  getOutputShape(input: UpdateValueInput): readonly number[] {
    return input.Value.shape;
  }

  compile(info: UpdateValueInfo, precision: Precision) {
    if (info.shapeMoments !== undefined) {
      this.maxRank = info.shapeMoments.length;
    }

    super.compile(info, precision);
  }

  getCompilationInfo(
    input: UpdateValueInput,
    precision: Precision
  ): UpdateValueInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      precision
    );

    return {
      shapeValue: input.Value.shape,
      widthValue: input.Value.memory.width,
      heightValue: input.Value.memory.height,

      shapeMoments: input.Moments.shape,
      widthMoments: input.Moments.memory.width,
      heightMoments: input.Moments.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      alpha: input.alpha,
      epsilon: input.epsilon,
    } as UpdateValueInfo;
  }

  getInputInfoString(input: UpdateValueInput): string {
    return `${input.Value.shape}-${input.Moments.shape}-${input.alpha}-${input.epsilon}`;
  }
}

export const defaultUpdateValueD = new Dispatcher(
  () => new UpdateValueOperation(gpuConstructor)
);
