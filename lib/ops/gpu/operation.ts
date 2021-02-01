import {DrawCommand, Framebuffer2D} from 'regl';
import {defaultAllocator, gl} from '../../tensor/gpu/gl';
import {GPUTensorConstructor, GPUTensorI} from '../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../tensor/gpu/memory';
import {Precision} from '../../types';
import {computeStrides, getSize} from '../../util/shape';

export const defaultMaxRank = 10;
export const defaultMaxIterations = 10000000;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type DictBase = {[name: string]: any};

type InputType = 'int' | 'float';

export interface Input {
  name: string;
  length?: number;
  type?: InputType;
}

export abstract class Operation<
  GPUTensor extends GPUTensorI,
  Info extends DictBase,
  InputType
> {
  protected allocator: GPUMemoryAllocator;

  protected statics: Set<string> = new Set<string>();

  protected gpuTensorConstructor: GPUTensorConstructor<GPUTensor>;

  protected maxRank: number;

  protected drawCommand?: DrawCommand;

  protected precision?: Precision;

  private copyCounter = 0;

  protected fullyStatic = false;
  protected outputShape?: readonly number[];

  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator,
    maxRank?: number
  ) {
    if (allocator === undefined) {
      allocator = defaultAllocator;
    }
    if (maxRank === undefined) {
      maxRank = defaultMaxRank;
    }
    this.allocator = allocator;
    this.maxRank = maxRank;

    this.gpuTensorConstructor = tensorConstructor;
  }

  registerStatics(info: Info) {
    let staticTextures = 0;
    let staticVars = 0;

    for (const key in info) {
      if (key.startsWith('shape')) {
        const texName = key.slice('shape'.length);
        this.statics.add(`shape${texName}`);
        this.statics.add(`size${texName}`);
        this.statics.add(`rank${texName}`);
        this.statics.add(`strides${texName}`);
        staticTextures++;
      } else {
        this.statics.add(key);
        if (!key.startsWith('width') && !key.startsWith('height')) {
          staticVars++;
        }
      }
    }

    if (
      staticTextures - 1 === this.getTextureNames().length &&
      staticVars === this.getUniformAttrs().length
    ) {
      this.fullyStatic = true;
      this.outputShape = info['shapeOutput'];
    }
  }

  getVarModifier(name: string) {
    return this.statics.has(name) ? '' : 'uniform';
  }

  pad(arr: number[], len = this.maxRank) {
    while (arr.length < len) {
      arr.push(-1);
    }
    return arr;
  }

  copyPad(arr: readonly number[], len = this.maxRank) {
    const result = Array.from(arr);
    while (result.length < len) {
      result.push(-1);
    }
    return result;
  }

  getVariables() {
    return '';
  }

  getVariableDeclarations() {
    const textures = this.getTextureNames();
    textures.push('Output');

    return `
      ${textures
        .map(x => {
          return `
        ${x === 'Output' ? '' : `uniform sampler2D ${x};`}
        ${this.getVarModifier('size' + x)} int size${x};
        ${this.getVarModifier('width' + x)} int width${x};
        ${this.getVarModifier('height' + x)} int height${x};
        ${this.getVarModifier('strides' + x)} int strides${x}[${this.maxRank}];
        ${this.getVarModifier('shape' + x)} int shape${x}[${this.maxRank}];
        ${this.getVarModifier('rank' + x)} int rank${x};
        `;
        })
        .join('\n')}
      varying vec2 uv;

      ${this.getVariables()}`;
  }

  getVariableInitializations(info: Info) {
    const textures = this.getTextureNames();
    textures.push('Output');

    let inits = '';
    for (const tex of textures) {
      if (`shape${tex}` in info) {
        const shape = info[`shape${tex}`] as number[];
        const strides = computeStrides(shape);
        const size = getSize(shape);
        const rank = shape.length;

        inits += this.getArrayInit(`shape${tex}`, shape);
        inits += this.getArrayInit(`strides${tex}`, strides);
        inits += `\nsize${tex} = ${size};`;
        inits += `\nrank${tex} = ${rank};`;
      }
    }
    for (const k in info) {
      if (!k.startsWith('shape')) {
        if (Array.isArray(info[k])) {
          inits += this.getArrayInit(k, info[k]);
        } else {
          const type = this.getVarType(k);

          if (type === 'int') {
            inits += `\n${k} = ${info[k]};`;
          } else {
            inits += `\n${k} = ${(info[k] as number).toPrecision(20)};`;
          }
        }
      }
    }

    return inits;
  }

  getVarType(name: string) {
    const res = this.getUniformAttrs().find(x => x.name === name);
    if (res !== undefined) {
      return res.type ? res.type : 'int';
    }
    return 'int';
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  getArrayInit(name: string, values: any[], len?: number, pad?: string) {
    if (len === undefined) {
      len = this.maxRank;
    }

    const type = this.getVarType(name);

    if (pad === undefined) {
      if (type === 'int') {
        pad = '-1';
      } else if (type === 'float') {
        pad = '-1.0';
      }
    }
    let res = '';
    for (let i = 0; i < len; i++) {
      if (i < values.length) {
        if (type === 'int') {
          res += `\n ${name}[${i}] = ${values[i]};`;
        } else if (type === 'float') {
          res += `\n ${name}[${i}] = ${(values[i] as number).toPrecision(20)};`;
        }
      } else {
        res += `\n ${name}[${i}] = ${pad};`;
      }
    }
    return res;
  }

  getUtilFunctions() {
    return `
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

    int indexToPos(int index[${this.maxRank}], int strides[${this.maxRank}]) {
      int pos = 0;
      for (int i = 0; i < ${this.maxRank}; i++) {
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

    float getValueAt(int index[${this.maxRank}], int strides[${this.maxRank}], int textureWidth, int textureHeight, sampler2D tex) {
      int pos = indexToPos(index, strides);
      return getValueAtPos(pos, textureWidth, textureHeight, tex);
    }`;
  }

  getTextureFunctions() {
    const textures = this.getTextureNames();

    return textures
      .map(x => {
        return `
      float _${x}(int indices[${this.maxRank}]) {
        return getValueAt(indices, strides${x}, width${x}, height${x}, ${x});
      }
      `;
      })
      .join('\n');
  }

  getCompleteFragmentShader(info: Info, precision: Precision): string {
    const fragShader = this.getFragmentShader(info);

    const variableDecls = this.getVariableDeclarations();
    const varInits = this.getVariableInitializations(info);

    const utilFunctions = this.getUtilFunctions();
    const textureFunctions = this.getTextureFunctions();

    const result = `
    precision ${this.precisionString(precision)} float;

    ${variableDecls}

    ${utilFunctions}
    ${textureFunctions}

    void initVars() {
      ${varInits}
    }

    ${fragShader}`;

    return result;
  }

  getUniforms(info: Info) {
    const uniformAttrs: Input[] = [];

    const defaultUniformAttrs: Input[] = this.getUniformAttrs();
    for (const defaultAttr of defaultUniformAttrs) {
      if (info[defaultAttr.name] === undefined) {
        uniformAttrs.push(defaultAttr);
      }
    }

    const textures = this.getTextureNames();
    textures.push('Output');
    for (const texture of textures) {
      uniformAttrs.push({name: texture});

      if (info[`shape${texture}`] === undefined) {
        uniformAttrs.push({name: `size${texture}`});
        uniformAttrs.push({name: `strides${texture}`, length: this.maxRank});
        uniformAttrs.push({name: `shape${texture}`, length: this.maxRank});
        uniformAttrs.push({name: `rank${texture}`});
      }
      if (info[`width${texture}`] === undefined) {
        uniformAttrs.push({name: `width${texture}`});
      }
      if (info[`height${texture}`] === undefined) {
        uniformAttrs.push({name: `height${texture}`});
      }
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const uniforms: any = {};
    for (const uniformAttr of uniformAttrs) {
      if (info[uniformAttr.name] === undefined) {
        if (uniformAttr.length !== undefined) {
          for (let i = 0; i < uniformAttr.length; i++) {
            const name = `${uniformAttr.name}[${i}]`;
            uniforms[name] = gl.prop(name as never);
          }
        } else {
          uniforms[uniformAttr.name] = gl.prop(uniformAttr.name as never);
        }
      }
    }

    return uniforms;
  }

  posToIndex(strides: string, result: string, pos: string) {
    const name = `${pos}_${this.copyCounter++}`;
    return `
    int ${name} = ${pos};
    for (int i = 0; i < ${this.maxRank}; i++) {
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
    }`;
  }

  initIndex(index: string, rank?: string) {
    if (rank === undefined) {
      return `
        for (int i = 0; i < ${this.maxRank}; i++) {
          ${index}[i] = -1;
        }`;
    } else {
      return `
        for (int i = 0; i < ${this.maxRank}; i++) {
          if (i < ${rank}) {
            ${index}[i] = 0;
          } else {
            ${index}[i] = -1;
          }
        }`;
    }
  }

  incrementIndex(index: string, shape: string) {
    return `
    for (int i = ${this.maxRank} - 1; i >= 0; i--) {
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

  incrementConditional(index: string, shape: string, cond: string) {
    return `
    for (int i = 0; i < ${this.maxRank}; i++) {
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

  getDefaultMain() {
    return `
    void main() {
      initVars();

      int pos = coordinateToPos(uv, widthOutput, heightOutput);

      vec4 result = vec4(0,0,0,0);

      if (pos < sizeOutput) {
        int index[${this.maxRank}];
        ${this.posToIndex('stridesOutput', 'index', 'pos')}
        result.r = process(index);

        pos += 1;

        if (pos < sizeOutput) {
          ${this.posToIndex('stridesOutput', 'index', 'pos')}
          result.g = process(index);

          pos += 1;

          if (pos < sizeOutput) {
            ${this.posToIndex('stridesOutput', 'index', 'pos')}
            result.b = process(index);

            pos += 1;

            if (pos < sizeOutput) {
              ${this.posToIndex('stridesOutput', 'index', 'pos')}
              result.a = process(index);
            }
          }
        }
      }

      gl_FragColor = result;
    }`;
  }

  precisionString(precision: Precision) {
    return precision === 32 ? 'highp' : 'mediump';
  }

  getDrawCommand(info: Info, precision: Precision): DrawCommand {
    const fragShader = this.getCompleteFragmentShader(info, precision);

    const uniforms = this.getUniforms(info);

    const result = gl({
      frag: fragShader,
      vert: `
        precision ${this.precisionString(precision)} float;
        attribute vec2 position;
        varying vec2 uv;
        void main() {
          uv = 0.5 * (position + 1.0);
          gl_Position = vec4(position, 0, 1);
        }`,
      attributes: {
        position: [-4, -4, 4, -4, 0, 4],
      },
      uniforms: uniforms,
      framebuffer: gl.prop('framebuffer' as never),
      depth: {
        enable: false,
      },
      count: 3,
    });

    return result;
  }

  compile(info: Info, precision: Precision) {
    this.registerStatics(info);

    this.precision = precision;

    this.drawCommand = this.getDrawCommand(info, precision);
  }

  compute(
    resultShape: readonly number[],
    inputTensors: {[name: string]: GPUTensorI},
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    inputs?: any
  ) {
    if (this.drawCommand === undefined) {
      this.compile({} as Info, 32);
    }

    const resultSize = getSize(resultShape);
    //@ts-ignore
    const result = this.allocator.allocate(resultSize, this.precision);

    const inputTextures: {[name: string]: Framebuffer2D} = {};
    for (const name in inputTensors) {
      inputTextures[name] = inputTensors[name].memory.frameBuffer;
    }
    if (inputs === undefined) {
      inputs = {};
    }

    if (!this.fullyStatic) {
      for (const name in inputTensors) {
        if (!this.statics.has(`shape${name}`)) {
          inputs[`size${name}`] = inputTensors[name].size;
          inputs[`strides${name}`] = this.pad(
            computeStrides(inputTensors[name].shape)
          );
          inputs[`shape${name}`] = this.copyPad(inputTensors[name].shape);
          inputs[`rank${name}`] = inputTensors[name].shape.length;
        }

        if (!this.statics.has(`width${name}`)) {
          inputs[`width${name}`] = inputTensors[name].memory.width;
        }

        if (!this.statics.has(`height${name}`)) {
          inputs[`height${name}`] = inputTensors[name].memory.height;
        }
      }

      if (!this.statics.has('shapeOutput')) {
        inputs['sizeOutput'] = resultSize;
        inputs['stridesOutput'] = this.pad(computeStrides(resultShape));
        inputs['shapeOutput'] = this.copyPad(resultShape);
        inputs['rankOutput'] = resultShape.length;
      }

      if (!this.statics.has('widthOutput')) {
        inputs['widthOutput'] = result.width;
      }

      if (!this.statics.has('heightOutput')) {
        inputs['heightOutput'] = result.height;
      }
    }

    //@ts-ignore
    this.drawCommand({
      framebuffer: result.frameBuffer,
      ...inputTextures,
      ...inputs,
    });

    //@ts-ignore
    return this.gpuTensorConstructor(result, resultShape, this.precision);
  }

  abstract getFragmentShader(info: Info): string;

  abstract getTextureNames(): string[];

  getUniformAttrs(): Input[] {
    return [];
  }

  abstract calc(input: InputType): GPUTensor;

  abstract getOutputShape(input: InputType): readonly number[];

  abstract getCompilationInfo(input: InputType, precision: Precision): Info;

  abstract getInputInfoString(input: InputType): string;
}
