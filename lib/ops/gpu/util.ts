import { DrawCommand, Framebuffer2D } from "regl";
import { defaultAllocator, gl } from "../../tensor/gpu/gl";
import { MemoryEntry } from "../../tensor/gpu/memory";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { getSize, computeStrides } from "../../util/shape";

export interface Input {
  name: string,
  length?: number
}

export const maxRank = 10;

export const maxIterations = 10000000;

export function pad(arr: number[], len=maxRank) {
  while (arr.length < len) {
    arr.push(-1);
  }
  return arr;
}

export function copyPad(arr: readonly number[], len=maxRank) {
  const result = Array.from(arr);
  while (result.length < len) {
    result.push(-1);
  }
  return result;
}

export const utilFunctions = `
int fromFloat(float f) {
  return int(floor(f+0.5));
}

int coordinateToPos(vec2 coordinate, int textureWidth, int textureHeight) {
  int x = (fromFloat(coordinate.x*float(textureWidth*2))-1)/2;
  int y = (fromFloat(coordinate.y*float(textureHeight*2))-1)/2;

  int pos = x + y*textureWidth;

  return pos*4;
}

vec2 posToCoordinate(int pos, int textureWidth, int textureHeight) {
  // 4 positions map to the same coordinate
  pos = pos/4;

  int y = pos/textureWidth;
  int x = pos - y*textureWidth;

  return vec2(float(x*2+1)/float(textureWidth*2), float(y*2+1)/float(textureHeight*2));
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

float getValueAtPos(int pos, int textureWidth, int textureHeight, sampler2D tex) {
  vec2 coord = posToCoordinate(pos, textureWidth, textureHeight);
  int res = pos - (pos/4)*4;
  vec4 val = texture2D(tex, coord);
  if (res == 0) {
    return val.r;
  } else if (res == 1) {
    return val.g;
  } else if (res == 2) {
    return val.b;
  } else {
    return val.a;
  }
}

float getValueAt(int index[${maxRank}], int strides[${maxRank}], int textureWidth, int textureHeight, sampler2D tex) {
  int pos = indexToPos(index, strides);
  return getValueAtPos(pos, textureWidth, textureHeight, tex);
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
      if (${strides}[i] == 0) {
        ${result}[i] = 0;
      } else {
        ${result}[i] = ${name}/${strides}[i];
        ${name} = ${name} - ${strides}[i]*${result}[i]; // Stupid modulo hack
      }
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
  for (int i = ${maxRank} - 1; i >= 0; i--) {
    if (${shape}[i] != -1) {
      ${index}[i] += 1;
      if (${index}[i] >= ${shape}[i]) {
        ${index}[i] = 0;
      } else {
        break;
      }
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
  int pos = coordinateToPos(uv, widthOutput, heightOutput);

  vec4 result = vec4(0,0,0,0);

  if (pos < sizeOutput) {
    int index[${maxRank}];
    ${posToIndex('stridesOutput', 'index', 'pos')}
    result.r = process(index);

    pos += 1;

    if (pos < sizeOutput) {
      ${posToIndex('stridesOutput', 'index', 'pos')}
      result.g = process(index);

      pos += 1;

      if (pos < sizeOutput) {
        ${posToIndex('stridesOutput', 'index', 'pos')}
        result.b = process(index);

        pos += 1;

        if (pos < sizeOutput) {
          ${posToIndex('stridesOutput', 'index', 'pos')}
          result.a = process(index);
        }
      }
    }
  }

  gl_FragColor = result;
}`;

function buildCompleteFragmentShader(fragmentShader: string, inputTextures: string[]) {
  return `
  precision highp float;
  ${inputTextures.map(x => {
    return `
    uniform sampler2D ${x};
    uniform int size${x};
    uniform int width${x};
    uniform int height${x};
    uniform int strides${x}[${maxRank}];
    uniform int shape${x}[${maxRank}];
    uniform int rank${x};
    `;
  }).join('\n')}
  uniform int sizeOutput;
  uniform int widthOutput;
  uniform int heightOutput;
  uniform int stridesOutput[${maxRank}];
  uniform int shapeOutput[${maxRank}];
  uniform int rankOutput;
  varying vec2 uv;

  ${utilFunctions}

  ${inputTextures.map(x => {
    return `
    float _${x}(int indices[${maxRank}]) {
      return getValueAt(indices, strides${x}, width${x}, height${x}, ${x});
    }
    `;
  }).join('\n')}

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
    uniform_attrs.push({name: `width${inputTexture}`});
    uniform_attrs.push({name: `height${inputTexture}`});
    uniform_attrs.push({name: `strides${inputTexture}`, length: maxRank});
    uniform_attrs.push({name: `shape${inputTexture}`, length: maxRank});
    uniform_attrs.push({name: `rank${inputTexture}`});
  }
  uniform_attrs.push({name: `sizeOutput`});
  uniform_attrs.push({name: `widthOutput`});
  uniform_attrs.push({name: `heightOutput`});
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
  const resultSize = getSize(resultShape);
  let result = defaultAllocator.allocate(resultSize);

  const inputTextures: {[name: string]: Framebuffer2D} = {};
  for (let name in inputTensors) {
    inputTextures[name] = inputTensors[name].memory.frameBuffer
  }
  if (inputs === undefined) {
    inputs = {};
  }
  for (let name in inputTensors) {
    inputs[`size${name}`] = inputTensors[name].size;
    inputs[`width${name}`] = inputTensors[name].memory.width;
    inputs[`height${name}`] = inputTensors[name].memory.height;
    inputs[`strides${name}`] = pad(computeStrides(inputTensors[name].getShape()));
    inputs[`shape${name}`] = copyPad(inputTensors[name].getShape());
    inputs[`rank${name}`] = inputTensors[name].getShape().length;
  }
  inputs['sizeOutput'] = resultSize;
  inputs['widthOutput'] = result.width;
  inputs['heightOutput'] = result.height;
  inputs['stridesOutput'] = pad(computeStrides(resultShape));
  inputs['shapeOutput'] = copyPad(resultShape);
  inputs['rankOutput'] = resultShape.length;

  op({
    framebuffer: result.frameBuffer,
    ...inputTextures,
    ...inputs
  });

  return new GPUTensor(result, resultShape);
}
