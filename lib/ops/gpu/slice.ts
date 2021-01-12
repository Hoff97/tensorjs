import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { buildComp, compute, defaultMain, initIndex, maxRank, pad } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform int offsets[${maxRank}];

float process(int index[${maxRank}]) {
  int inIx[${maxRank}];
  ${initIndex('inIx')}
  for (int i = 0; i < ${maxRank}; i++) {
    if (index[i] == -1) {
      break;
    }

    inIx[i] = index[i] + offsets[i];
  }

  return _x(inIx);
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['x'], fragmentShader, [
    {name: 'offsets', length: maxRank},
  ]);
}

export function slice(x: GPUTensor, starts: number[], ends: number[], axes: number[]) {
  if (comp === undefined) {
    initComp();
  }

  const rank = x.shape.length;

  const resultShape = [...x.shape];
  const offsets: number[] = new Array(rank).fill(0);
  let axIx = 0;
  for (let i = 0; i < rank && axIx < axes.length; i++) {
    if (i == axes[axIx]) {
      resultShape[i] = ends[axIx] - starts[axIx];
      offsets[i] = starts[axIx];
      axIx++;
    }
  }

  return compute(comp, resultShape, {
    x: x
  }, {
    offsets: pad(offsets)
  });
}