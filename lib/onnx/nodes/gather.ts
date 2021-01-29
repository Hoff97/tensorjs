import { GatherInfo, GatherOperation } from "../../ops/gpu/gather";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class GatherNode extends OnnxNode {
  private axis: number;

  private compiled = false;
  private operation?: GatherOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axis = this.getAttributeInt("axis") || 0;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    const indices = inputs[1];

    if (!(indices instanceof CPUTensor)) {
      throw new Error("Gather requires CPU tensor for the indices");
    }

    if (!this.compiled) {
      return [x.gather(this.axis, indices)];
    } else {
      return [this.operation.calc({
        X: x as GPUTensor,
        axis: this.axis,
        indices
      })];
    }
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    const x = inputs[0];
    const indices = inputs[1];

    if (!(indices instanceof CPUTensor)) {
      throw new Error("Gather requires CPU tensor for the indices");
    }

    const resultShape = this.operation.getOutputShape({X: x as any, axis: this.axis, indices});
    const memory = this.allocator.allocate(getSize(resultShape), precision);

    if (compile) {
      const xMem = (x as any).memory;

      const info: GatherInfo = {
        shapeX: x.getShape(),
        widthX: xMem.width,
        heightX: xMem.height,

        shapeOutput: resultShape,
        widthOutput: memory.width,
        heightOutput: memory.height,

        axis: this.axis,
        indices: indices
      };

      this.operation.compile(info, precision);

      this.compiled = true;
    }

    return {
      outputs: [new PrototypeTensor(resultShape, memory)]
    };
  }

  initializeForCompiling(): void {
    this.operation = new GatherOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Gather';
  }

  delete(): void {}
}