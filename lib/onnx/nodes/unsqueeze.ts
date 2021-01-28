import { CopyInfo, CopyOperation } from "../../ops/gpu/copy";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class UnsqueezeNode extends OnnxNode {
  private axes: number[];

  private compiled = false;
  private operation?: CopyOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 13) {
      this.axes = this.getAttributeInts("axes");
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (this.onnxVersion < 13) {
      const currShape = x.getShape();
      const newShape = [];
      let axIx = 0;
      for (let i = 0; i < currShape.length; i++) {
        if (axIx < this.axes.length && this.axes[axIx] === i) {
          newShape.push(1);
          axIx++;
        }
        newShape.push(currShape[i]);
      }
      if (this.axes[this.axes.length - 1] === currShape.length) {
        newShape.push(1);
      }

      if (!this.compiled) {
        return [x.reshape(newShape)];
      } else {
        return [this.operation.calc({
          input: x as GPUTensor,
          outputShape: newShape
        })];
      }
    }
    throw new Error(`Unsqueeze with onnx version ${this.onnxVersion} not yet implemented`);
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 13) {
      const x = inputs[0];

      const resultShape = [];
      let axIx = 0;
      for (let i = 0; i < x.getShape().length; i++) {
        if (axIx < this.axes.length && this.axes[axIx] === i) {
          resultShape.push(1);
          axIx++;
        }
        resultShape.push(x.getShape()[i]);
      }
      if (this.axes[this.axes.length - 1] === x.getShape().length) {
        resultShape.push(1);
      }

      const memory = this.allocator.allocate(getSize(resultShape), precision);

      if (compile) {
        const [xMem] = this.getMemoryEntries(inputs);

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
    throw new Error(`Unsqueeze is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.operation = new CopyOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Unsqueeze';
  }
}