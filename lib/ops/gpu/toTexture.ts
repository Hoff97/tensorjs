import { DrawCommand } from "regl";
import { defaultAllocator } from "../../tensor/gpu/gl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { computeStrides } from "../../util/shape";
import { buildComp, copyPad, initIndex, maxRank, pad } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform int nChannels;
void main() {
  int x = (fromFloat(uv.x*float(widthOutput*2))-1)/2;
  int y = (fromFloat(uv.y*float(heightOutput*2))-1)/2;

  int inIx[${maxRank}];
  ${initIndex('inIx')}

  inIx[0] = y;
  inIx[1] = x;
  inIx[2] = 0;

  float a = _inputTensor(inIx);
  float b = 0.0;
  float c = 0.0;
  float d = 255.0;

  if (nChannels >= 2) {
    inIx[2] = 1;
    b = _inputTensor(inIx);
  }
  if (nChannels >= 3) {
    inIx[2] = 2;
    c = _inputTensor(inIx);
  }
  if (nChannels >= 4) {
    inIx[2] = 3;
    d = _inputTensor(inIx);
  }

  gl_FragColor = vec4(a,b,c,d);
}`;

function initComp() {
  comp = buildComp(['inputTensor'], fragmentShader, [{name: 'nChannels'}]);
}

export function toTexture(tensor: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  return compute(tensor);
}

export function compute(input: GPUTensor) {
    let result = defaultAllocator.allocateOfDimensions(input.shape[1], input.shape[0]);

    const inputs: any = {};
    inputs[`sizeinputTensor`] = input.size;
    inputs[`widthinputTensor`] = input.memory.width;
    inputs[`heightinputTensor`] = input.memory.height;
    inputs[`stridesinputTensor`] = pad(computeStrides(input.shape));
    inputs[`shapeinputTensor`] = copyPad(input.shape);
    inputs[`rankinputTensor`] = input.shape.length;

    const resultShape = [input.shape[0], input.shape[1], 4]

    inputs['sizeOutput'] = result.size;
    inputs['widthOutput'] = result.width;
    inputs['heightOutput'] = result.height;
    inputs['stridesOutput'] = pad(computeStrides(resultShape));
    inputs['shapeOutput'] = copyPad(resultShape);
    inputs['rankOutput'] = resultShape.length;

    comp({
      framebuffer: result.frameBuffer,
      inputTensor: input.memory.frameBuffer,
      nChannels: input.shape[2],
      ...inputs
    });

    return new GPUTensor(result, resultShape);
}
