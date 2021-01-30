import { SliceInfo, SliceOperation } from "../../ops/gpu/util/slice";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class SliceNode extends OnnxNode {
  private axes?: number[];
  private starts: number[];
  private ends: number[];

  private compiled = false;
  private operation?: SliceOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axes = this.getAttributeInts("axes");
    this.starts = this.getAttributeInts("starts");
    this.ends = this.getAttributeInts("ends");
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      const x = inputs[0];
      if (!this.compiled) {
        return [x.slice(this.starts, this.ends, this.axes)];
      } else {
        const axes = this.getAxes(x);
        return [this.operation.calc({
          X: x as GPUTensor, axes,
          starts: this.starts,
          ends: this.ends
        })];
      }
    }
    throw new Error(`Slice not implemented for onnx version ${this.onnxVersion}`);
  }

  getAxes(x: Tensor) {
    if (this.axes !== undefined) {
      return this.axes;
    }
    const axes = new Array(x.getShape().length);
    for (let i = 0; i < x.getShape().length; i++) {
      axes[i] = i;
    }
    return axes;
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 11) {
      const x = inputs[0];
      const axes = this.getAxes(x);

      for (let i = 0; i < axes.length; i++) {
        const sh = x.getShape()[axes[i]];
        if (this.starts[i] < 0) {
          this.starts[i] += sh;
        } else if (this.starts[i] >= sh) {
          this.starts[i] = sh;
        }
        if (this.ends[i] < 0) {
          this.ends[i] += sh;
        } else if (this.ends[i] >= sh) {
          this.ends[i] = sh;
        }
      }

      const resultShape = this.operation.getOutputShape({X: x as GPUTensor, axes, ends: this.ends, starts: this.starts});

      const memory = this.allocator.allocate(getSize(resultShape), precision);

      if (compile) {
        const [xMem] = this.getMemoryEntries(inputs);

        const info: SliceInfo = {
          shapeX: x.getShape(),
          widthX: xMem.width,
          heightX: xMem.height,

          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,

          axes,
          starts: this.starts,
          ends: this.ends
        };

        this.operation.compile(info, precision);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`Slice is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.operation = new SliceOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Slice';
  }

  delete(): void {}
}