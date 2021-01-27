import { ExpandInfo, ExpandOperation } from "../../ops/gpu/expand";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";
import { createTensor } from "../util";

export class ExpandNode extends OnnxNode {
  private compiled = false;
  private operation?: ExpandOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 13) {
      let tensor = inputs[0];

      const _shape = inputs[1];
      if (!(_shape instanceof CPUTensor)) {
        throw new Error("Expand needs cpu tensor as shape tensor");
      }
      const shape = new Array(_shape.size);
      for (let i = 0; i < _shape.size; i++) {
        shape[i] = _shape.get(i);
      }

      if (!this.compiled) {
        return [tensor.expand(shape)];
      } else {
        const [xShapeAligned, _, resultShape] = tensor.alignShapes(tensor.getShape(), shape);

        tensor = tensor.reshape(xShapeAligned, false);

        return [this.operation.calc({
          input: tensor as GPUTensor,
          outputShape: resultShape
        })];
      }
    }
    throw new Error(`Expand not yet implemented for onnx version ${this.onnxVersion}`);
  }

  async staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 13) {
      const x = inputs[0];

      const _shape = inputs[1];
      if (!(_shape instanceof CPUTensor)) {
        throw new Error("Expand needs cpu tensor as shape tensor");
      }
      const shape = new Array(_shape.size);
      for (let i = 0; i < _shape.size; i++) {
        shape[i] = _shape.get(i);
      }

      const [xShapeAligned, _, resultShape] = x.alignShapes(x.getShape(), shape);

      const memory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const xMem = (x as any).memory;

        const info: ExpandInfo = {
          shapeX: xShapeAligned,
          widthX: xMem.width,
          heightX: xMem.height,

          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,
        };

        this.operation.compile(info);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`Expand not yet implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.operation = new ExpandOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Expand';
  }
}