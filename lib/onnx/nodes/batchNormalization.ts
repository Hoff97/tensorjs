import { AddOperation } from "../../ops/gpu/add";
import { MultiplyOperation } from "../../ops/gpu/multiply";
import { NormalizeOperation, NormalizeOpInfo } from "../../ops/gpu/normalize";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { glContext } from "../../tensor/gpu/gl";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { toCPU, toGPU, toWASM } from "../../util/convert";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class BatchNormalizationNode extends OnnxNode {
  private epsilon: number;
  private momentum: number;

  private epsTensor: Tensor;

  private scale?: Tensor;
  private bias?: Tensor;

  private compiled = false;

  private normOperation?: NormalizeOperation<GPUTensor>;

  private addOperation?: AddOperation<GPUTensor>;
  private multiplyOperation?: MultiplyOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.epsilon = this.getAttributeFloat('epsilon') || 1e-05;
    this.momentum = this.getAttributeFloat('momentum') || 0.9;

    this.epsTensor = new CPUTensor([1], [this.epsilon]);

    //TODO: Handle lower onnxversions here
  }

  initialize(resolveConstant: (name: string) => Tensor) {
    const scale = resolveConstant(this.inputs[1]);
    const B = resolveConstant(this.inputs[2]);
    const mean = resolveConstant(this.inputs[3]);
    const variance = resolveConstant(this.inputs[4]);

    if (scale !== undefined && B !== undefined && mean !== undefined && variance !== undefined) {
      const varSqrt = variance.add(this.epsTensor).sqrt();

      this.scale = scale.divide(varSqrt);
      this.bias = B.subtract(mean.multiply(this.scale));

      varSqrt.delete();
    }
  }

  async defaultForward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (this.scale !== undefined) {
      const scaled = x.multiply(this.scale);
      const result = scaled.add(this.bias);
      scaled.delete();

      return [result];
    } else {
      let scale = inputs[1];
      let B = inputs[2];
      let mean = inputs[3];
      let variance = inputs[4];

      //TODO: Handle lower onnx versions here

      const C = scale.getShape()[0];

      const newShape = [1,C,...new Array(x.getShape().length - 2).fill(1)];

      scale = scale.reshape(newShape, false);
      B = B.reshape(newShape, false);
      mean = mean.reshape(newShape, false);
      variance = variance.reshape(newShape, false);

      const result = x.normalize(mean, variance, this.epsilon, scale, B);

      return [result];
    }
  }

  async compiledForward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (this.scale !== undefined) {
      const scaled = this.multiplyOperation.calc({
        A: this.scale as GPUTensor,
        B: x as GPUTensor,
        outputShape: x.getShape()
      });
      const result = this.addOperation.calc({
        A: this.bias as GPUTensor,
        B: scaled,
        outputShape: x.getShape()
      });
      scaled.delete(this.allocator);

      return [result];
    } else {
      let scale = inputs[1];
      let B = inputs[2];
      let mean = inputs[3];
      let variance = inputs[4];

      const C = scale.getShape()[0];

      const newShape = [1,C,...new Array(x.getShape().length - 2).fill(1)];

      scale = scale.reshape(newShape, false);
      B = B.reshape(newShape, false);
      mean = mean.reshape(newShape, false);
      variance = variance.reshape(newShape, false);

      const result = this.normOperation.calc({
        X: x as GPUTensor,
        Mean: mean as GPUTensor,
        Variance: variance as GPUTensor,
        Scale: scale as GPUTensor,
        Bias: B as GPUTensor,
        epsilon: this.epsilon
      });

      return [result];
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (!this.compiled) {
      return this.defaultForward(inputs);
    } else {
      return this.compiledForward(inputs);
    }
  }

  async staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    const x = inputs[0];

    const resultShape = x.getShape();

    let resultMemory;

    if (this.bias !== undefined) {
      const intermediateMemory = this.allocator.allocate(getSize(resultShape));
      resultMemory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const [xMem, scaleMem, biasMem, meanMem, varianceMem] = this.getMemoryEntries(inputs);

        let scale = inputs[1];
        const C = scale.getShape()[0];
        const newShape = [1,C,...new Array(x.getShape().length - 2).fill(1)];

        this.scale = this.scale.reshape(newShape, false);
        this.bias = this.bias.reshape(newShape, false);

        const memScale = (this.scale as GPUTensor).memory;
        const memBias = (this.bias as GPUTensor).memory;

        this.multiplyOperation.compile({
          shapeA: newShape,
          widthA: memScale.width,
          heightA: memScale.height,

          shapeB: x.getShape(),
          widthB: xMem.width,
          heightB: xMem.height,

          shapeOutput: x.getShape(),
          widthOutput: intermediateMemory.width,
          heightOutput: intermediateMemory.height
        });

        this.addOperation.compile({
          shapeA: newShape,
          widthA: memBias.width,
          heightA: memBias.height,

          shapeB: x.getShape(),
          widthB: intermediateMemory.width,
          heightB: intermediateMemory.height,

          shapeOutput: x.getShape(),
          widthOutput: resultMemory.width,
          heightOutput: resultMemory.height
        });

        this.compiled = true;
      }

      this.allocator.deallocate(intermediateMemory);

    } else {
      resultMemory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const [xMem, scaleMem, biasMem, meanMem, varianceMem] = this.getMemoryEntries(inputs);

        let scale = inputs[1];
        const C = scale.getShape()[0];
        const newShape = [1,C,...new Array(x.getShape().length - 2).fill(1)];

        const info: NormalizeOpInfo = {
          shapeX: x.getShape(),
          widthX: xMem.width,
          heightX: xMem.height,

          shapeBias: newShape,
          heightBias: biasMem.height,
          widthBias: biasMem.width,

          shapeMean: newShape,
          heightMean: meanMem.height,
          widthMean: meanMem.width,

          shapeScale: newShape,
          heightScale: scaleMem.height,
          widthScale: scaleMem.width,

          shapeVariance: newShape,
          heightVariance: varianceMem.height,
          widthVariance: varianceMem.width,

          shapeOutput: resultShape,
          widthOutput: resultMemory.width,
          heightOutput: resultMemory.height,

          epsilon: this.epsilon
        };

        this.normOperation.compile(info);

        this.compiled = true;
      }
    }

    return {
      outputs: [new PrototypeTensor(resultShape, resultMemory)]
    };
  }

  initializeForCompiling(): void {
    if (this.bias !== undefined) {
      this.addOperation = new AddOperation(gpuConstructor, this.allocator);
      this.multiplyOperation = new MultiplyOperation(gpuConstructor, this.allocator);
    } else {
      this.normOperation = new NormalizeOperation(gpuConstructor, this.allocator);
    }
  }

  async toCPU() {
    if (this.scale !== undefined) {
      this.scale = await toCPU(this.scale);
      this.bias = await toCPU(this.bias);
    }
  }

  async toWASM() {
    if (this.scale !== undefined) {
      this.scale = await toWASM(this.scale);
      this.bias = await toWASM(this.bias);
    }
  }

  async toGPU() {
    if (this.scale !== undefined) {
      this.scale = await toGPU(this.scale);
      this.bias = await toGPU(this.bias);
    }
  }
}