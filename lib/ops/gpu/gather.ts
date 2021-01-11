import { DrawCommand } from "regl";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { computeStrides } from "../../util/shape";
import { buildComp, compute, copyPad, defaultMain, maxRank, pad } from "./util";

let comp: DrawCommand;

const gatherMaxIxSize = 10;

const fragmentShader = `
uniform int axis;
uniform int indexValues[${gatherMaxIxSize}];

uniform int mappedIndexStrides[${maxRank}];
uniform int mappedInputStrides[${maxRank}];

float process(int index[${maxRank}]) {
  int inputPos = 0;
  int indexPos = 0;

  int strideAxis = 0;
  for (int i = 0; i < ${maxRank}; i++) {
    if (index[i] == -1) {
      break;
    }
    if (i == axis) {
      strideAxis = stridesx[i];
    }
    inputPos += mappedInputStrides[i]*index[i];
    indexPos += mappedIndexStrides[i]*index[i];
  }

  for (int i = 0; i < ${gatherMaxIxSize}; i++) {
    if (i == indexPos) {
      inputPos += indexValues[i]*strideAxis;
      break;
    }
  }

  return getValueAtPos(inputPos, widthx, heightx, x);
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['x'], fragmentShader, [
    {name: 'axis'},
    {name: 'indexValues', length: gatherMaxIxSize},
    {name: 'mappedInputStrides', length: maxRank},
    {name: 'mappedIndexStrides', length: maxRank}
  ]);
}

export function gather(x: GPUTensor, axis: number, indices: CPUTensor) {
  if (indices.size > gatherMaxIxSize) {
    throw new Error(`Gather on GPU can deal with at most ${gatherMaxIxSize} indices, input had ${indices.size}`);
  }

  if (comp === undefined) {
    initComp();
  }

  const r = x.shape.length;
  const q = indices.shape.length;

  const inputStrides = computeStrides(x.shape);
  const indexStrides = computeStrides(indices.shape);

  const resultRank = r + q - 1;
  const resultShape = new Array(resultRank);

  const mappedInputStrides = new Array(resultRank).fill(0);
  const mappedIndexStrides = new Array(resultRank).fill(0);

  for (let i = 0; i < axis; i++) {
    resultShape[i] = x.shape[i];
    mappedInputStrides[i] = inputStrides[i];

    mappedIndexStrides[i] = 0;
  }
  for (let i = 0; i < q; i++) {
    resultShape[i + axis] = indices.shape[i];
    mappedIndexStrides[i + axis] = indexStrides[i];

    mappedInputStrides[i + axis] = 0;
  }
  for (let i = axis + 1; i < r; i++) {
    resultShape[i + q - 1] = x.shape[i];
    mappedInputStrides[i + q - 1] = inputStrides[i];

    mappedIndexStrides[i + q - 1] = 0;
  }

  return compute(comp, resultShape, {
    x: x
  }, {
    axis,
    indexValues: pad(Array.from(indices.values), gatherMaxIxSize),
    mappedInputStrides: pad(mappedInputStrides),
    mappedIndexStrides: pad(mappedIndexStrides)
  });
}