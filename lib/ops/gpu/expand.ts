import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { buildComp, compute, defaultMain, maxRank } from "./util";

let comp: DrawCommand;

const fragmentShader = `
float process(int index[${maxRank}]) {
  return _x(index);
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['x'], fragmentShader);
}

export function expand(tensor: GPUTensor, resultShape: readonly number[]) {
  if (comp === undefined) {
    initComp();
  }

  return compute(comp, resultShape, {
    x: tensor,
  });
}