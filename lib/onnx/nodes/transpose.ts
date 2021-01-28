import { TransposeInfo, TransposeOperation } from "../../ops/gpu/transpose";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class TransposeNode extends OnnxNode {
  private permutation?: number[];

  private compiled = false;
  private operation?: TransposeOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.permutation = this.getAttributeInts("perm");
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const a = inputs[0];

    if (!this.compiled) {
      return [a.transpose(this.permutation)];
    } else {
      return [this.operation.calc({A: a as GPUTensor, permutation: this.getPermutation(a)})];
    }
  }

  getPermutation(x: Tensor) {
    if (this.permutation !== undefined) {
      return this.permutation;
    }
    const rank = x.getShape().length;
    const perm = new Array(rank);
    for (let i = 0; i < rank; i++) {
      perm[i] = rank - i - 1;
    }
    return perm;
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    const x = inputs[0];
    const permutation = this.getPermutation(x);

    const resultShape = this.operation.getOutputShape({A: x as any, permutation});
    const memory = this.allocator.allocate(getSize(resultShape), precision);

    if (compile) {
      const [xMem] = this.getMemoryEntries(inputs);

      const info: TransposeInfo = {
        shapeA: x.getShape(),
        widthA: xMem.width,
        heightA: xMem.height,

        shapeOutput: resultShape,
        widthOutput: memory.width,
        heightOutput: memory.height,

        permutation
      };

      this.operation.compile(info, precision);

      this.compiled = true;
    }

    return {
      outputs: [new PrototypeTensor(resultShape, memory)]
    };
  }

  initializeForCompiling(): void {
    this.operation = new TransposeOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Transpose';
  }
}