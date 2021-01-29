import { CopyInfo, CopyOperation } from "../../ops/gpu/copy";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class ReshapeNode extends OnnxNode {
  private shape: number[];

  private compiled = false;
  private operation?: CopyOperation<GPUTensor>;
  private resultShape?: readonly number[];

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 13) {
      this.shape = this.getAttributeInts("shape");
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    const shape = inputs[1];

    if (!(shape instanceof CPUTensor)) {
      throw new Error('Reshape only works with CPU tensor as shape tensor');
    }

    if (this.onnxVersion < 13) {
      const _shape = new Array(shape.size);
      for (let i = 0; i < shape.size; i++) {
        _shape[i] = shape.get(i);
      }

      if (!this.compiled) {
        return [x.reshape(_shape)];
      } else {
        return [this.operation.calc({input: x as GPUTensor, outputShape: this.resultShape})];
      }
    }
    throw new Error(`Reshape with onnx version ${this.onnxVersion} not yet implemented`);
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 13) {
      const x = inputs[0];
      const shape = inputs[1];

      if (!(shape instanceof CPUTensor)) {
        throw new Error('Reshape only works with CPU tensor as shape tensor');
      }
      const resultShape = new Array(shape.size);
      let negativeIx = -1;
      let outSize = 1;
      for (let i = 0; i < shape.size; i++) {
        resultShape[i] = shape.get(i);
        if (shape.get(i) < 0) {
          negativeIx = i;
        } else {
          outSize *= shape.get(i);
        }
      }
      if (negativeIx > 0) {
        const currSize = getSize(x.getShape());
        resultShape[negativeIx] = currSize/outSize;
      }

      this.resultShape = resultShape;

      const memory = this.allocator.allocate(getSize(resultShape), precision);

      if (compile) {
        const xMem = (x as any).memory;

        const info: CopyInfo = {
          shapeX: x.getShape(),
          widthX: xMem.width,
          heightX: xMem.height,

          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,
        };

        this.operation.compile(info, precision);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`Reshape is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.operation = new CopyOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Reshape';
  }

  delete(): void {}
}