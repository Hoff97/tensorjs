import { UpsampleInfo, UpsampleOperation } from "../../ops/gpu/upsample";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { toCPU } from "../../util/convert";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class UpsampleNode extends OnnxNode {
  private mode: string;

  private operation?: UpsampleOperation<GPUTensor>;
  private compiled = false;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.mode = this.getAttributeString("mode");

    if (this.mode !== "nearest") {
      throw new Error("Upsampling only supported with nearest neighbor sampling");
    }
  }

  async getScales(scale: Tensor) {
    if (!(scale instanceof CPUTensor)) {
      console.warn("Scales tensor for upsample not on CPU, need to transfer!");
      scale = await toCPU(scale);
    }

    const sc = scale as CPUTensor;

    const scales = new Array(sc.size);
    for (let i = 0; i < sc.size; i++) {
      scales[i] = sc.get(i);
    }
    return scales;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    let scale = inputs[1];

    const scales = await this.getScales(scale);

    if (!this.compiled) {
      return [x.upsample(scales)];
    } else {
      return [this.operation.calc({
        X: x as GPUTensor,
        scales: scales
      })];
    }
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    const x = inputs[0];

    let scale = inputs[1];
    const scales = await this.getScales(scale);

    const resultShape = this.operation.getOutputShape({
      X: x as any, scales
    })

    const memory = this.allocator.allocate(getSize(resultShape), precision);

    if (compile) {
      const xMem = (x as any).memory;

      const info: UpsampleInfo = {
        shapeX: x.getShape(),
        widthX: xMem.width,
        heightX: xMem.height,
        shapeOutput: resultShape,
        widthOutput: memory.width,
        heightOutput: memory.height,
        scales
      };

      this.operation.compile(info, precision);

      this.compiled = true;
    }

    return {
      outputs: [new PrototypeTensor(resultShape, memory)]
    };
  }

  initializeForCompiling(): void {
    this.operation = new UpsampleOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Upsample';
  }

  delete(): void {}
}