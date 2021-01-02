import { DrawCommand, Framebuffer2D } from "regl";
import GPUTensor, { gl } from "../../tensor/gpu/tensor";
import { getSize, computeStrides } from "../../util/shape";

export interface Input {
  name: string,
  length?: number
}

export const maxRank = 10;

export const maxIterations = 10000000;

export function pad(arr: number[]) {
  while (arr.length < maxRank) {
    arr.push(-1);
  }
  return arr;
}

export function copyPad(arr: readonly number[]) {
  const result = Array.from(arr);
  while (result.length < maxRank) {
    result.push(-1);
  }
  return result;
}

export const utilFunctions = `
int coordinateToPos(float coordinate, int size) {
  return int(coordinate*float(size)) - 2;
}

float posToCoordinate(int pos, int size) {
  // 4 positions map to the same coordinate
  pos = (pos/4)*4+2;
  float coordinate = float(pos)/float(size);
  return coordinate;
}

int indexToPos(int index[${maxRank}], int strides[${maxRank}]) {
  int pos = 0;
  for (int i = 0; i < ${maxRank}; i++) {
    if (strides[i] == -1) {
      break;
    }
    pos += index[i]*strides[i];
  }
  return pos;
}

float getValueAt(int index[${maxRank}], int strides[${maxRank}], int size, sampler2D tex) {
  int pos = indexToPos(index, strides);
  float coord = posToCoordinate(pos, size);
  int res = pos - (pos/4)*4;
  vec4 val = texture2D(tex, vec2(coord, 0.5));
  if (res == 0) {
    return val.r;
  } else if (res == 1) {
    return val.g;
  } else if (res == 2) {
    return val.b;
  } else {
    return val.a;
  }
}`;

let copyCounter = 0;

export function posToIndex(strides: string, result: string, pos: string) {
  const name = `${pos}_${copyCounter++}`;
  return `
  int ${name} = ${pos};
  for (int i = 0; i < ${maxRank}; i++) {
    if (${strides}[i] == -1) {
      ${result}[i] = -1;
    } else {
      ${result}[i] = ${name}/${strides}[i];
      ${name} = ${name} - ${strides}[i]*${result}[i]; // Stupid modulo hack
    }
  }`
}

export function initIndex(index: string, rank?: string) {
  if (rank === undefined) {
    return `
      for (int i = 0; i < ${maxRank}; i++) {
        ${index}[i] = -1;
      }`;
  } else {
    return `
      for (int i = 0; i < ${maxRank}; i++) {
        if (i < ${rank}) {
          ${index}[i] = 0;
        } else {
          ${index}[i] = -1;
        }
      }`;
  }
}

export function incrementIndex(index: string, shape: string) {
  return `
  for (int i = 0; i < ${maxRank}; i++) {
    ${index}[i] += 1;
    if (${index}[i] >= ${shape}[i]) {
      ${index}[i] = 0;
    } else {
      break;
    }
  }
  `;
}

export function incrementConditional(index: string, shape: string, cond: string) {
  return `
  for (int i = 0; i < ${maxRank}; i++) {
    if (${cond}[i] == 1) {
      ${index}[i] += 1;
      if (${index}[i] >= ${shape}[i]) {
        ${index}[i] = 0;
      } else {
        break;
      }
    } else if (${cond}[i] == -1) {
      break;
    }
  }
  `;
}

export const defaultMain = `
void main() {
  int pos = coordinateToPos(uv.x, sizeOutput);

  int index[${maxRank}];
  ${posToIndex('stridesOutput', 'index', 'pos')}
  float a = process(index);

  pos += 1;
  ${posToIndex('stridesOutput', 'index', 'pos')}
  float b = process(index);

  pos += 1;
  ${posToIndex('stridesOutput', 'index', 'pos')}
  float c = process(index);

  pos += 1;
  ${posToIndex('stridesOutput', 'index', 'pos')}
  float d = process(index);

  gl_FragColor = vec4(a, b, c, d);
}`;

function buildCompleteFragmentShader(fragmentShader: string, inputTextures: string[]) {
  return `
  precision highp float;
  ${inputTextures.map(x => {
    return `
    uniform sampler2D ${x};
    uniform int size${x};
    uniform int strides${x}[${maxRank}];
    uniform int shape${x}[${maxRank}];
    uniform int rank${x};
    `;
  }).join('\n')}
  uniform int sizeOutput;
  uniform int stridesOutput[${maxRank}];
  uniform int shapeOutput[${maxRank}];
  uniform int rankOutput;
  varying vec2 uv;

  ${utilFunctions}

  ${fragmentShader}
  `
}

export function buildComp(inputTextures: string[], fragmentShader: string,
                          uniform_attrs?: Input[]) {
  if (uniform_attrs === undefined) {
    uniform_attrs = [];
  }

  const uniforms: any = {};
  for (let inputTexture of inputTextures) {
    uniforms[inputTexture] = gl.prop(inputTexture as never);

    uniform_attrs.push({name: `size${inputTexture}`});
    uniform_attrs.push({name: `strides${inputTexture}`, length: maxRank});
    uniform_attrs.push({name: `shape${inputTexture}`, length: maxRank});
    uniform_attrs.push({name: `rank${inputTexture}`});
  }
  uniform_attrs.push({name: `sizeOutput`});
  uniform_attrs.push({name: `stridesOutput`, length: maxRank});
  uniform_attrs.push({name: `shapeOutput`, length: maxRank});
  uniform_attrs.push({name: `rankOutput`});

  for (let uniform_attr of uniform_attrs) {
    if (uniform_attr.length !== undefined) {
      for (let i = 0; i < uniform_attr.length; i++) {
        const name = `${uniform_attr.name}[${i}]`
        uniforms[name] = gl.prop(name as never);
      }
    } else {
      uniforms[uniform_attr.name] = gl.prop(uniform_attr.name as never);
    }
  }

  const result = gl({
    frag: buildCompleteFragmentShader(fragmentShader, inputTextures),
    vert: `
      precision highp float;
      attribute vec2 position;
      varying vec2 uv;
      void main() {
        uv = 0.5 * (position + 1.0);
        gl_Position = vec4(position, 0, 1);
      }`,
    attributes: {
      position: [-4, -4, 4, -4, 0, 4]
    },
    uniforms: uniforms,
    framebuffer: gl.prop('framebuffer' as never),
    depth: {
      enable: false
    },
    count: 3
  });
  return result;
}

export function compute(op: DrawCommand,
                        resultShape: readonly number[],
                        inputTensors: {[name: string]: GPUTensor},
                        inputs?: {[name: string]: any}) {
  const inputTextures: {[name: string]: Framebuffer2D} = {};
  for (let name in inputTensors) {
    inputTextures[name] = inputTensors[name].framebuffer
  }
  if (inputs === undefined) {
    inputs = {};
  }
  for (let name in inputTensors) {
    inputs[`size${name}`] = inputTensors[name].size;
    inputs[`strides${name}`] = pad(computeStrides(inputTensors[name].getShape()));
    inputs[`shape${name}`] = copyPad(inputTensors[name].getShape());
    inputs[`rank${name}`] = inputTensors[name].getShape().length;
  }
  inputs['sizeOutput'] = Math.ceil(getSize(resultShape) / 4)*4;
  inputs['stridesOutput'] = pad(computeStrides(resultShape));
  inputs['shapeOutput'] = copyPad(resultShape);
  inputs['rankOutput'] = resultShape.length;

  const resultSize = getSize(resultShape);
  const textureSize = Math.ceil(resultSize / 4);
  const result = gl.framebuffer({
    width: textureSize,
    height: 1,
    depthStencil: false,
    colorFormat: 'rgba',
    colorType: 'float'
  });

  op({
    framebuffer: result,
    ...inputTextures,
    ...inputs
  });

  return new GPUTensor(result, resultShape);
}