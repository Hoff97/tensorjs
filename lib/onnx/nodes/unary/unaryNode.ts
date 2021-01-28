import { UnaryOperation } from "../../../ops/gpu/unaryOperation";
import { PrototypeTensor } from "../../../tensor/cpu/prototype";
import { CPUTensor } from "../../../tensor/cpu/tensor";
import { GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../../types";
import { getSize } from "../../../util/shape";
import { OnnxNode } from "../../node";
import { Attributes, Constants } from "../../types";

export abstract class UnaryNode extends OnnxNode {
  protected operation?: UnaryOperation<GPUTensor>;

  protected compiled: boolean = false;

  protected name: string;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  abstract compute(x: Tensor): Tensor;

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (!this.compiled) {
      return [this.compute(x)];
    } else {
      return [this.operation.calc({input: x as GPUTensor})];
    }
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs as CPUTensor[]);
    }

    const x = inputs[0];

    const memory = this.allocator.allocate(getSize(x.getShape()), precision);

    if (compile) {
      const [xMem] = this.getMemoryEntries(inputs);

      const info = {
        shapeX: x.getShape(),
        widthX: xMem.width,
        heightX: xMem.height,
        shapeOutput: x.getShape(),
        widthOutput: memory.width,
        heightOutput: memory.height
      };

      this.operation.compile(info, precision);

      this.compiled = true;
    }

    return {
      outputs: [new PrototypeTensor(x.getShape(), memory)]
    };
  }

  abstract getOperation(): UnaryOperation<GPUTensor>;

  initializeForCompiling(): void {
    this.operation = this.getOperation();
  }

  getType() {
    return this.name;
  }
}