import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Input, Operation } from "./operation";


export interface ClipInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;

  minVal?: number;
  maxVal?: number;
  doMin?: number;
  doMax?: number;
}

export interface ClipInput {
  input: GPUTensorI;
  minVal?: number;
  maxVal?: number;
}

export class ClipOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, ClipInfo, ClipInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: ClipInfo): string {
    return `
    void main() {
      initVars();

      vec4 maxVec = vec4(maxVal,maxVal,maxVal,maxVal);
      vec4 minVec = vec4(minVal,minVal,minVal,minVal);

      vec4 res = texture2D(X, uv);
      if (doMin == 1) {
        res = max(minVec, res);
      }
      if (doMax == 1) {
        res = min(maxVec, res);
      }

      gl_FragColor = res;
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  getVariables() {
    return `
    ${this.getVarModifier('minVal')} float minVal;
    ${this.getVarModifier('maxVal')} float maxVal;
    ${this.getVarModifier('doMin')} int doMin;
    ${this.getVarModifier('doMax')} int doMax;
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "minVal", type: "float" },
      { name: "maxVal", type: "float" },
      { name: "doMin"},
      { name: "doMax"}
    ];
  }

  calc(input: ClipInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input}, {
      minVal: input.minVal !== undefined ? input.minVal : 0,
      maxVal: input.maxVal !== undefined ? input.maxVal : 0,
      doMin: input.minVal !== undefined ? 1 : 0,
      doMax: input.maxVal !== undefined ? 1 : 0
    });
  }

  getOutputShape(input: ClipInput): readonly number[] {
    return input.input.shape;
  }

  compile(info: ClipInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }
}
