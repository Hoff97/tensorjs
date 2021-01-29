import { ConvBiasOperation, ConvInfo, ConvInput, ConvOperation } from "../../ops/gpu/conv";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { MemoryEntry } from "../../tensor/gpu/memory";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Activation, Precision } from "../../types";
import { toCPU, toGPU, toWASM } from "../../util/convert";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class ConvNode extends OnnxNode {
  private group: number;
  private dilations?: number[];
  private pads?: number[];
  private strides?: number[];

  private compiled = false;

  private operation?: ConvOperation<GPUTensor>;

  public kernel?: Tensor;
  public bias?: Tensor;

  private activation: Activation;

  constructor(attributes: Attributes, inputs: string[],
              outputs: string[], constants: Constants,
              onnxVersion: number,
              kernel?: Tensor,
              bias?: Tensor,
              activation?: Activation) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    const autoPad = this.getAttributeString('autoPad');
    if (autoPad !== undefined) {
      throw new Error('Autopad in conv not supported yet');
    }

    if (activation === undefined) {
      activation = "id";
    }
    this.activation = activation;

    this.group = this.getAttributeInt('group') || 1;
    this.dilations = this.getAttributeInts('dilations');
    this.pads = this.getAttributeInts('pads');
    this.strides = this.getAttributeInts('strides');

    this.kernel = kernel;
    this.bias = bias;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    const w = this.kernel !== undefined ? this.kernel : inputs[1];
    const b = inputs.length > 2 ? inputs[2] : this.bias;

    if (!this.compiled) {
      return [x.conv(w, b, this.dilations, this.group, this.pads, this.strides, this.activation)];
    } else {
      const rank = inputs[0].getShape().length - 2;
      const dilations = this.getDilations(rank);
      const pads = this.getPads(rank);
      const strides = this.getStrides(rank);

      let input: ConvInput = {
        X: x as GPUTensor,
        W: w as GPUTensor,
        pads, dilations, strides,
        activation: this.activation
      };

      if (b !== undefined) {
        //@ts-ignore
        input['B'] = b;
      }

      return [this.operation.calc(input)];
    }
  }

  getDilations(rank: number) {
    if (this.dilations !== undefined) {
      return this.dilations;
    }
    return new Array(rank).fill(1);
  }

  getPads(rank: number) {
    if (this.pads !== undefined) {
      return this.pads;
    }
    return new Array(rank*2).fill(0);
  }

  getStrides(rank: number) {
    if (this.strides !== undefined) {
      return this.strides;
    } return new Array(rank).fill(1);
  }

  getMemoryEntries(inputs: Tensor[]): MemoryEntry[] {
    const memories = super.getMemoryEntries(inputs);
    if (this.kernel === undefined) {
      return memories;
    } else {
      memories.push((this.kernel as GPUTensor).memory);
      if (this.bias !== undefined) {
        memories.push((this.bias as GPUTensor).memory);
      }
    }
    return memories;
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.operation === undefined) {
      this.initializeForCompiling(inputs);
    }

    const x = inputs[0];
    const w = this.kernel !== undefined ? this.kernel : inputs[1];
    const bias = inputs.length > 2 ? inputs[2] : this.bias;

    const rank = x.getShape().length - 2;
    const dilations = this.getDilations(rank);
    const pads = this.getPads(rank);
    const strides = this.getStrides(rank);

    const resultShape = this.operation.getOutputShape({
      X: x as any,
      W: w as any,
      pads, dilations, strides,
      activation: this.activation
    });
    const memory = this.allocator.allocate(getSize(resultShape), precision);

    if (compile) {
      const memories = this.getMemoryEntries(inputs);
      const memX = memories[0];
      const memW = memories[1];

      let info: ConvInfo = {
        shapeX: x.getShape(),
        widthX: memX.width,
        heightX: memX.height,
        shapeW: w.getShape(),
        widthW: memW.width,
        heightW: memW.height,
        shapeOutput: resultShape,
        widthOutput: memory.width,
        heightOutput: memory.height,

        activation: this.activation,

        pads, dilations, strides
      };

      if (bias !== undefined) {
        const memB = memories[2];

        info = {
          ...info,
          shapeB: bias.getShape(),
          heightB: memB.height,
          widthB: memB.width
        } as any;
      }

      this.operation.compile(info, precision);

      this.compiled = true;
    }

    return {
      outputs: [new PrototypeTensor(resultShape, memory)]
    };
  }

  initializeForCompiling(inputs?: Tensor[]): void {
    if (inputs !== undefined) {
      if (this.kernel === undefined) {
        if (inputs.length === 2) {
          this.operation = new ConvOperation(gpuConstructor, this.allocator);
        } else if(inputs.length === 3) {
          this.operation = new ConvBiasOperation(gpuConstructor, this.allocator);
        }
      } else {
        if (this.bias === undefined) {
          this.operation = new ConvOperation(gpuConstructor, this.allocator);
        } else {
          this.operation = new ConvBiasOperation(gpuConstructor, this.allocator);
        }
      }
    }
  }

  getType() {
    return 'Conv';
  }

  async toCPU() {
    if (this.kernel !== undefined) {
      this.kernel = await toCPU(this.kernel);
    }
    if (this.bias !== undefined) {
      this.bias = await toCPU(this.bias);
    }
  }

  async toWASM() {
    if (this.kernel !== undefined) {
      this.kernel = await toWASM(this.kernel);
    }
    if (this.bias !== undefined) {
      this.bias = await toWASM(this.bias);
    }
  }

  async toGPU(precision: Precision) {
    if (this.kernel !== undefined) {
      this.kernel = await toGPU(this.kernel, precision);
    }
    if (this.bias !== undefined) {
      this.bias = await toGPU(this.bias, precision);
    }
  }


  delete(): void {
    if (this.kernel !== undefined) {
      this.kernel.delete();
    }
    if (this.bias !== undefined) {
      this.bias.delete();
    }
  }
}
