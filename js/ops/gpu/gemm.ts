import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { buildComp, compute, maxRank, defaultMain, initIndex, maxIterations } from "./util";

let comp: DrawCommand;
let compWithC: DrawCommand;

const variables = `
uniform int M;
uniform int N;
uniform int O;
uniform int rank;
uniform int aTranspose;
uniform int bTranspose;
uniform float alpha;
uniform float beta;
`;

const body = `
int ixA[${maxRank}];
  ${initIndex('ixA')}
  int ixB[${maxRank}];
  ${initIndex('ixA')}
  for (int i = 0; i < ${maxRank}; i++) {
    if (i >= rank - 2) {
      break;
    }
    ixA[i] = index[i];
    ixB[i] = index[i];
  }

  int m = 0;
  int o = 0;
  for (int i = 0; i < ${maxRank}; i++) {
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

  for (int n = 0; n < ${maxIterations}; n++) {
    if (n >= N) {
      break;
    }
    for (int i = 0; i < ${maxRank}; i++) {
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
    res += _a(ixA) * _b(ixB);
  }

  res = res*alpha;
`;

const fragmentShader = `
${variables}

float process(int index[${maxRank}]) {
  ${body}

  return res;
}

${defaultMain}
`;

const fragmentShaderWithC = `
${variables}

float process(int index[${maxRank}]) {
  ${body}

  res += beta*_c(index);

  return res;
}

${defaultMain}
`;

function initComp() {
  const uniforms = [
    {name: 'M'}, {name: 'N'}, {name: 'O'}, {name: 'rank'},
    {name: 'aTranspose'}, {name: 'bTranspose'}, {name: 'alpha'}, {name: 'beta'}
  ];
  comp = buildComp(['a', 'b'], fragmentShader, uniforms);
  compWithC = buildComp(['a', 'b', 'c'], fragmentShaderWithC, uniforms);
}

export function gemm(a: GPUTensor, b: GPUTensor, aTranspose: boolean, bTranspose: boolean,
                     alpha: number, beta: number, c?: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  const rank = a.shape.length;

  const M = aTranspose ? a.shape[rank - 1] : a.shape[rank - 2];
  const N = aTranspose ? a.shape[rank - 2] : a.shape[rank - 1];
  const O = bTranspose ? b.shape[rank - 2] : b.shape[rank - 1];

  const batchShape = a.shape.slice(0, rank-2);
  const resultShape = [...batchShape, M, O];

  const uniforms = {
    M, N, O, rank,
    aTranspose: aTranspose ? 1 : 0,
    bTranspose: bTranspose ? 1 : 0,
    alpha, beta
  };

  if (c !== undefined) {
    return compute(compWithC, resultShape, {
      a, b, c
    }, uniforms);
  } else {
    return compute(comp, resultShape, {
      a, b
    }, uniforms);
  }
}