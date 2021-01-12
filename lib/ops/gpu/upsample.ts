import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { buildComp, compute, copyPad, defaultMain, initIndex, maxRank, pad } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform float scales[${maxRank}];

float process(int index[${maxRank}]) {
  int inIx[${maxRank}];
  ${initIndex('inIx')}

  for (int i = 0; i < ${maxRank}; i++) {
    if (index[i] == -1) {
      break;
    }

    inIx[i] = int(floor(float(index[i]) / scales[i]));
  }

  return _x(inIx);
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['x'], fragmentShader, [
    {name: 'scales', length: maxRank},
  ]);
}

export function upsample(x: GPUTensor, scales: number[]) {
  if (comp === undefined) {
    initComp();
  }

  const rank = x.shape.length;

  const resultShape = [...x.shape];
  for (let i = 0; i < rank; i++) {
    resultShape[i] = Math.floor(resultShape[i] * scales[i]);
  }

  return compute(comp, resultShape, {
    x: x
  }, {
    scales: copyPad(scales)
  });
}