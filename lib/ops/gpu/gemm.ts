import { getSize } from "../../util/shape";
import { outputDimsSize } from "../util/conv";

import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Input, Operation } from "./operation";
import { Precision } from "../../types";
import { defaultAllocator } from "../../tensor/gpu/gl";


export interface GemmInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;
  shapeB?: readonly number[];
  widthB?: number;
  heightB?: number;
  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;

  aTranspose?: boolean | number;
  bTranspose?: boolean | number;
  alpha?: number;
  beta?: number;

  M?: number;
  N?: number;
  O?: number;

  rank?: number;
}

export interface GemmInput {
  a: GPUTensorI;
  b: GPUTensorI;
  aTranspose: boolean;
  bTranspose: boolean;
  alpha: number;
  beta: number;
}

export class GemmOperation<GPUTensor extends GPUTensorI, GemmInf extends GemmInfo = GemmInfo, GemmIn extends GemmInput = GemmInput> extends Operation<GPUTensor, GemmInf, GemmIn> {
  protected maxIterations = 1000000;

  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getMainBody() {
    return `
      int ixA[${this.maxRank}];
      ${this.initIndex('ixA')}
      int ixB[${this.maxRank}];
      ${this.initIndex('ixA')}
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (i >= rank - 2) {
          break;
        }
        ixA[i] = index[i];
        ixB[i] = index[i];
      }

      int m = 0;
      int o = 0;
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (i == rank-2) {
          m = index[i];
          o = index[i+1];

          if (aTranspose == 0) {
            ixA[i] = m;
          } else {
            ixA[i+1] = m;
          }

          if (bTranspose == 0) {
            ixB[i+1] = o;
          } else {
            ixB[i] = o;
          }

          break;
        }
      }

      float res = 0.0;

      for (int n = 0; n < ${this.maxIterations}; n++) {
        if (n >= N) {
          break;
        }
        for (int i = 0; i < ${this.maxRank}; i++) {
          if (i == rank-2) {
            if (aTranspose == 0) {
              ixA[i+1] = n;
            } else {
              ixA[i] = n;
            }

            if (bTranspose == 0) {
              ixB[i] = n;
            } else {
              ixB[i+1] = n;
            }

            break;
          }
        }
        res += _A(ixA) * _B(ixB);
      }

      res = res*alpha;
    `;
  }

  getVariables() {
    return `
    ${this.getVarModifier('M')} int M;
    ${this.getVarModifier('N')} int N;
    ${this.getVarModifier('O')} int O;
    ${this.getVarModifier('rank')} int rank;
    ${this.getVarModifier('aTranspose')} int aTranspose;
    ${this.getVarModifier('bTranspose')} int bTranspose;
    ${this.getVarModifier('alpha')} float alpha;
    ${this.getVarModifier('beta')} float beta;
    `;
  }

  getFragmentShader(info: GemmInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      ${this.getMainBody()}

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["A", "B"];
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "M" },
      { name: "N" },
      { name: "O" },
      { name: "rank" },
      { name: "aTranspose" },
      { name: "bTranspose" },
      { name: "alpha", type: "float" },
      { name: "beta", type: "float" }
    ];
  }

  calc(input: GemmInput): GPUTensor {
    const rank = input.a.shape.length;

    const M = input.aTranspose ? input.a.shape[rank - 1] : input.a.shape[rank - 2];
    const N = input.aTranspose ? input.a.shape[rank - 2] : input.a.shape[rank - 1];
    const O = input.bTranspose ? input.b.shape[rank - 2] : input.b.shape[rank - 1];

    const batchShape = input.a.shape.slice(0, rank-2);
    const resultShape = [...batchShape, M, O];

    const uniforms = {
      M, N, O, rank,
      aTranspose: input.aTranspose ? 1 : 0,
      bTranspose: input.bTranspose ? 1 : 0,
      alpha: input.alpha,
      beta: input.beta
    };

    return this.compute(resultShape, {A: input.a, B: input.b}, uniforms);
  }

  getOutputShape(input: GemmIn): readonly number[] {
    const rank = input.a.shape.length;

    const M = input.aTranspose ? input.a.shape[rank - 1] : input.a.shape[rank - 2];
    const O = input.bTranspose ? input.b.shape[rank - 2] : input.b.shape[rank - 1];

    const batchShape = input.a.shape.slice(0, rank-2);
    const resultShape = [...batchShape, M, O];

    return resultShape;
  }

  compile(info: GemmInf, precision: Precision) {
    if (info.shapeA !== undefined) {
      const rank = info.shapeA.length;
      info.rank = rank;
      this.maxRank = rank;

      if (info.aTranspose !== undefined) {
        const M = info.aTranspose ? info.shapeA[rank - 1] : info.shapeA[rank - 2];
        const N = info.aTranspose ? info.shapeA[rank - 2] : info.shapeA[rank - 1];

        info.M = M;
        info.N = N;

        info.aTranspose = info.aTranspose ? 1 : 0;
      }
    }

    if (info.shapeB !== undefined && info.bTranspose !== undefined) {
      const rank = info.shapeB.length;
      const O = info.bTranspose ? info.shapeB[rank - 2] : info.shapeB[rank - 1];

      info.O = O;
      info.bTranspose = info.bTranspose ? 1 : 0;
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: GemmIn, precision: Precision): GemmInf {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(getSize(outputShape), precision);

    const rank = input.a.shape.length;

    const M = input.aTranspose ? input.a.shape[rank - 1] : input.a.shape[rank - 2];
    const N = input.aTranspose ? input.a.shape[rank - 2] : input.a.shape[rank - 1];
    const O = input.bTranspose ? input.b.shape[rank - 2] : input.b.shape[rank - 1];

    const info: GemmInfo = {
      shapeA: input.a.shape,
      widthA: input.a.memory.width,
      heightA: input.a.memory.height,

      shapeB: input.b.shape,
      widthB: input.b.memory.width,
      heightB: input.b.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      M, N, O,

      aTranspose: input.aTranspose ? 1 : 0,
      bTranspose: input.bTranspose ? 1 : 0,
      alpha: input.alpha,
      beta: input.beta,

      rank
    };

    return info as GemmInf;
  }
}

export interface GemmCInfo extends GemmInfo {
  shapeC?: readonly number[];
  widthC?: number;
  heightC?: number;
}

export interface GemmCInput extends GemmInput {
  c: GPUTensorI;
}

export class GemmCOperation<GPUTensor extends GPUTensorI> extends GemmOperation<GPUTensor, GemmCInfo, GemmCInput> {
  getTextureNames(): string[] {
    return ["A", "B", "C"];
  }

  getFragmentShader(info: GemmInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      ${this.getMainBody()}

      res += beta*_C(index);

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  calc(input: GemmCInput): GPUTensor {
    const rank = input.a.shape.length;

    const M = input.aTranspose ? input.a.shape[rank - 1] : input.a.shape[rank - 2];
    const N = input.aTranspose ? input.a.shape[rank - 2] : input.a.shape[rank - 1];
    const O = input.bTranspose ? input.b.shape[rank - 2] : input.b.shape[rank - 1];

    const batchShape = input.a.shape.slice(0, rank-2);
    const resultShape = [...batchShape, M, O];

    const uniforms = {
      M, N, O, rank,
      aTranspose: input.aTranspose ? 1 : 0,
      bTranspose: input.bTranspose ? 1 : 0,
      alpha: input.alpha,
      beta: input.beta
    };

    return this.compute(resultShape, {A: input.a, B: input.b, C: input.c}, uniforms);
  }

  getCompilationInfo(input: GemmCInput, precision: Precision): GemmCInfo {
    const inf = super.getCompilationInfo(input, precision);

    const info: GemmCInfo = {
      ...inf,

      shapeC: input.c.shape,
      widthC: input.c.memory.width,
      heightC: input.c.memory.height
    };

    return info;
  }
}
