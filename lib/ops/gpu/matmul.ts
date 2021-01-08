import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { buildComp, compute, maxRank, defaultMain, initIndex, maxIterations } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform int k;

float process(int index[${maxRank}]) {
  int ix1[${maxRank}];
  ${initIndex('ix1')}
  ix1[0] = index[0];

  int ix2[${maxRank}];
  ${initIndex('ix2')}
  ix2[1] = index[1];

  float res = 0.0;

  for (int i = 0; i < ${maxIterations}; i++) {
    if (i >= k) {
      break;
    }
    ix1[1] = i;
    ix2[0] = i;

    float v1 = _input1(ix1);
    float v2 = _input2(ix2);
    res += v1*v2;
  }

  return res;
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['input1', 'input2'], fragmentShader, [{name: 'k'}]);
}

export function matmul(tensor1: GPUTensor, tensor2: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  const outputShape = [tensor1.getShape()[0], tensor2.getShape()[1]]

  return compute(comp, outputShape, {
    input1: tensor1,
    input2: tensor2
  }, {
    k: tensor1.getShape()[1]
  });
}