import { BinaryOperation } from "../../ops/gpu/binaryOperation";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export abstract class BinaryNode extends OnnxNode {
  private operation?: BinaryOperation<GPUTensor>;

  private compiled = false;

  protected name: string;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  abstract compute(a: Tensor, b: Tensor): Tensor;

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 13 && this.onnxVersion >= 7) {
      const a = inputs[0];
      const b = inputs[1];

      if (!this.compiled) {
        return [this.compute(a,b)];
      } else {
        const [_a, _b, resultShape] = a.alignTensor(b);

        return [this.operation.calc({A: _a as GPUTensor, B: _b as GPUTensor, outputShape: resultShape as number[]})];
      }
    }
    throw new Error(`${this.name} not implemented for onnx version ${this.onnxVersion}`);
  }

  async staticForward(inputs: Tensor[], compile?: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]}> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs as CPUTensor[]);
    }

    if (this.onnxVersion < 13 && this.onnxVersion >= 7) {
      const a = inputs[0];
      const b = inputs[1];

      const [_a,_b,resultShape] = a.alignShapes(a.getShape(), b.getShape());

      const memory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const [aMem, bMem] = this.getMemoryEntries(inputs);

        const info = {
          shapeA: _a,
          widthA: aMem.width,
          heightA: aMem.height,
          shapeB: _b,
          widthB: bMem.width,
          heightB: bMem.height,
          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height
        };

        this.operation.compile(info);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error("Method not implemented.");
  }

  abstract getOperation(): BinaryOperation<GPUTensor>;

  initializeForCompiling(): void {
    this.operation = this.getOperation();
  }
}