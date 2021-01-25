import { GemmInfo, GemmOperation } from "../../ops/gpu/gemm";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class MatMulNode extends OnnxNode {
  private compiled = false;
  private operation?: GemmOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const A = inputs[0];
    const B = inputs[1];

    if (this.onnxVersion < 13 && this.onnxVersion >= 9) {
      if (A.getShape().length !== B.getShape().length) {
        throw new Error("Automatic broadcasting in MatMul not supported yet");
      }

      if (!this.compiled) {
        return [A.gemm(B)];
      } else {
        return [this.operation.calc({a: A as GPUTensor, b: B as GPUTensor, aTranspose: false, bTranspose: false, alpha: 1, beta: 1})];
      }
    }
    throw new Error(`Matmul with onnx version ${this.onnxVersion} not yet implemented`);
  }

  async staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 13 && this.onnxVersion >= 9) {
      const a = inputs[0];
      const b = inputs[1];

      const resultShape = this.operation.getOutputShape({a: a as any, b: b as any, aTranspose: false, bTranspose: false, alpha: 1, beta: 1});
      const memory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const [aMem, bMem] = this.getMemoryEntries(inputs);

        const info: GemmInfo = {
          shapeA: a.getShape(),
          widthA: aMem.width,
          heightA: aMem.height,

          shapeB: b.getShape(),
          widthB: bMem.width,
          heightB: bMem.height,

          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,

          aTranspose: false,
          bTranspose: false,
          alpha: 1,
          beta: 1
        };

        this.operation.compile(info);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`MatMul is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.operation = new GemmOperation(gpuConstructor, this.allocator);
  }
}