import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { computeStrides, getSize } from "../../util/shape";
import { buildComp, compute, maxRank, utilFunctions, pad, posToIndex } from "./util";

let comp: DrawCommand;

const fragmentShader = `
float process(int index[${maxRank}]) {
  //return getValueAt(index, stridesOutput, sizeOutput, inputTensor2);
  return 1.0;
}

void main() {
  int pos = coordinateToPos(uv.x, sizeOutput);

  int index[${maxRank}];
  ${posToIndex('stridesOutput', 'index', 'pos')}
  float a = process(index);

  pos++;
  ${posToIndex('stridesOutput', 'index', 'pos')}
  float b = process(index);

  pos++;
  ${posToIndex('stridesOutput', 'index', 'pos')}
  float c = process(index);

  pos++;
  ${posToIndex('stridesOutput', 'index', 'pos')}
  float d = process(index);

  gl_FragColor = vec4(a, b, c, d);
}
`;

function initComp() {
  comp = buildComp(['input1', 'input2'], fragmentShader);
}

export function matmul(tensor1: GPUTensor, tensor2: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  return compute(comp, tensor1.getShape(), {
    inputTensor1: tensor1,
    inputTensor2: tensor2
  });
}