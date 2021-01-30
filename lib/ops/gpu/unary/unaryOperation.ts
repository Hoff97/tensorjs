import { defaultAllocator } from "../../../tensor/gpu/gl";
import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { Precision } from "../../../types";
import { getSize } from "../../../util/shape";
import { Operation } from "../operation";


export interface UnaryOpInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;
}

export interface UnaryOpInput {
  input: GPUTensorI;
}

export abstract class UnaryOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, UnaryOpInfo, UnaryOpInput> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  abstract operation(input: string): string;

  getFragmentShader(info: UnaryOpInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = ${this.operation('texture2D(X, uv)')};
    }
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  calc(input: UnaryOpInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input})
  }

  getOutputShape(input: UnaryOpInput): readonly number[] {
    return input.input.shape;
  }

  compile(info: UnaryOpInfo, precision: Precision) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: UnaryOpInput, precision: Precision): UnaryOpInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(getSize(outputShape), precision);

    return {
      shapeX: input.input.shape,
      widthX: input.input.memory.width,
      heightX: input.input.memory.height,
      shapeOutput: this.getOutputShape(input),
      widthOutput: outputSize.width,
      heightOutput: outputSize.height
    };
  }
}
