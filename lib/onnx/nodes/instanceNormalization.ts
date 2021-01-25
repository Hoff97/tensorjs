import { BinaryOpInfo } from "../../ops/gpu/binaryOperation";
import { NormalizeOperation, NormalizeOpInfo } from "../../ops/gpu/normalize";
import { PoolInfo } from "../../ops/gpu/pool";
import { ReduceMeanOperation } from "../../ops/gpu/reduceMean";
import { ReduceMeanSquareOperation } from "../../ops/gpu/reduceMeanSquare";
import { SubtractOperation } from "../../ops/gpu/subtract";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { glContext } from "../../tensor/gpu/gl";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { toCPU, toGPU, toWASM } from "../../util/convert";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class InstanceNormalizationNode extends OnnxNode {
  private epsilon: number;

  private compiled = false;

  private reduceMeanOperation?: ReduceMeanOperation<GPUTensor>;
  private subtractOperation?: SubtractOperation<GPUTensor>;
  private reduceMeanSquareOperation?: ReduceMeanSquareOperation<GPUTensor>;
  private normalizeOperation?: NormalizeOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.epsilon = this.getAttributeFloat('epsilon') || 1e-05;

    //TODO: Handle onnx versions < 6 here
  }

  async defaultForward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    let scale = inputs[1];
    let B = inputs[2];

    const dataRank = x.getShape().length - 2;

    const C = scale.getShape()[0];

    const newShape = [1,C,...new Array(dataRank).fill(1)];

    scale = scale.reshape(newShape, false);
    B = B.reshape(newShape, false);

    const reduceAxes = new Array(x.getShape().length  -2);
    for (let i = 0; i < dataRank; i++) {
      reduceAxes[i] = i+2;
    }

    const mean = x.reduceMean(reduceAxes, true);
    glContext.flush();
    const diff = x.subtract(mean);
    glContext.flush();
    const variance = diff.reduceMeanSquare(reduceAxes, true);
    glContext.flush();

    const result = x.normalize(mean, variance, this.epsilon, scale, B);

    mean.delete();
    diff.delete();
    variance.delete();

    return [result];
  }

  async compiledForward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    let scale = inputs[1];
    let B = inputs[2];

    const dataRank = x.getShape().length - 2;
    const C = scale.getShape()[0];
    const newShape = [1,C,...new Array(dataRank).fill(1)];

    scale = scale.reshape(newShape, false);
    B = B.reshape(newShape, false);

    const reduceAxes = new Array(x.getShape().length  -2);
    for (let i = 0; i < dataRank; i++) {
      reduceAxes[i] = i+2;
    }

    const mean = this.reduceMeanOperation.calc({X: x as GPUTensor, axes: reduceAxes, keepDims: true});
    glContext.flush();
    const diff = this.subtractOperation.calc({A: x as GPUTensor, B: mean, outputShape: x.getShape()});
    glContext.flush();
    const variance = this.reduceMeanSquareOperation.calc({X: diff, axes: reduceAxes, keepDims: true});
    glContext.flush();

    const result = this.normalizeOperation.calc({
      X: x as GPUTensor,
      Mean: mean,
      Variance: variance,
      Bias: B as GPUTensor,
      Scale: scale as GPUTensor,
      epsilon: this.epsilon
    });

    mean.delete();
    diff.delete();
    variance.delete();

    return [result];
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

    if (this.onnxVersion < 11) {
      const x = inputs[0];
      let scale = inputs[1];
      let B = inputs[2];

      const dataRank = x.getShape().length - 2;
      const C = scale.getShape()[0];
      const newShape = [1,C,...new Array(dataRank).fill(1)];

      const reduceAxes = new Array(x.getShape().length  -2);
      const reduceShape = [...x.getShape()];
      for (let i = 0; i < dataRank; i++) {
        reduceAxes[i] = i+2;
        reduceShape[i+2] = 1;
      }

      const resultShape = x.getShape();

      const meanMemory = this.allocator.allocate(getSize(reduceShape));
      const diffMemory = this.allocator.allocate(getSize(x.getShape()));
      const varianceMemory = this.allocator.allocate(getSize(reduceShape));
      const memory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const [xMem, scaleMem, biasMem] = this.getMemoryEntries(inputs);

        const meanInfo: PoolInfo = {
          shapeX: x.getShape(),
          widthX: xMem.width,
          heightX: xMem.height,

          shapeOutput: reduceShape,
          widthOutput: meanMemory.width,
          heightOutput: meanMemory.height,

          axes: reduceAxes, keepDims: true
        };
        this.reduceMeanOperation.compile(info);

        const diffInfo: BinaryOpInfo = {
          shapeA: x.getShape(),
          widthA: xMem.width,
          heightA: xMem.height,

          shapeB: reduceShape,
          widthB: meanMemory.width,
          heightB: meanMemory.height,

          shapeOutput: x.getShape(),
          widthOutput: diffMemory.width,
          heightOutput: diffMemory.height
        };
        this.subtractOperation.compile(diffInfo);

        const varInfo: PoolInfo = {
          shapeX: x.getShape(),
          widthX: diffMemory.width,
          heightX: diffMemory.height,

          shapeOutput: reduceShape,
          widthOutput: varianceMemory.width,
          heightOutput: varianceMemory.height,

          axes: reduceAxes, keepDims: true
        };
        this.reduceMeanSquareOperation.compile(varInfo);

        const normInfo: NormalizeOpInfo = {
          shapeX: x.getShape(),
          widthX: xMem.width,
          heightX: xMem.height,

          shapeMean: reduceShape,
          widthMean: meanMemory.width,
          heightMean: meanMemory.height,

          shapeVariance: reduceShape,
          widthVariance: varianceMemory.width,
          heightVariance: varianceMemory.height,

          shapeScale: newShape,
          widthScale: scaleMem.width,
          heightScale: scaleMem.height,

          shapeBias: newShape,
          widthBias: biasMem.width,
          heightBias: biasMem.height,

          shapeOutput: x.getShape(),
          widthOutput: memory.width,
          heightOutput: memory.height,

          epsilon: this.epsilon
        };
        this.normalizeOperation.compile(normInfo);

        this.compiled = true;
      }

      this.allocator.deallocate(meanMemory);
      this.allocator.deallocate(diffMemory);
      this.allocator.deallocate(varianceMemory);

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`Slice is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.reduceMeanOperation = new ReduceMeanOperation(gpuConstructor, this.allocator);
    this.subtractOperation = new SubtractOperation(gpuConstructor, this.allocator);
    this.reduceMeanSquareOperation = new ReduceMeanSquareOperation(gpuConstructor, this.allocator);
    this.normalizeOperation = new NormalizeOperation(gpuConstructor, this.allocator);
  }
}