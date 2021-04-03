import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {gpuConstructor, GPUTensor} from '../../../tensor/gpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {compareShapes, computeStrides, getSize} from '../../../util/shape';
import {Dispatcher} from '../../gpu/dispatcher';
import {Input, Operation} from '../../gpu/operation';

export interface ReshapeIndicesInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  nnzFraction?: number;
  sparseDims?: number;
  oldSparseStrides?: number[];
  newSparseStrides?: number[];
}

export interface ReshapeIndicesInput {
  A: GPUTensorI;
  sparseShape: readonly number[];
  shape: readonly number[];
  nnz: number;
}

export class ReshapeIndicesOperation<
  GPUTensor extends GPUTensorI
> extends Operation<GPUTensor, ReshapeIndicesInfo, ReshapeIndicesInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('nnzFraction')} int nnzFraction;
    ${this.getVarModifier('sparseDims')} int sparseDims;
    ${this.getVarModifier('oldSparseStrides')} int oldSparseStrides[${
      this.maxRank
    }];
    ${this.getVarModifier('newSparseStrides')} int newSparseStrides[${
      this.maxRank
    }];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'nnzFraction'},
      {name: 'sparseDims'},
      {name: 'oldSparseStrides', length: this.maxRank},
      {name: 'newSparseStrides', length: this.maxRank},
    ];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: ReshapeIndicesInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      float res = 0.0;

      int newNNZ = int(index[0]);
      int oldNnzIx = newNNZ / nnzFraction;
      int residualNNZ = newNNZ - oldNnzIx*nnzFraction;

      int oldSparseIx[${this.maxRank}];
      ${this.initIndex('oldSparseIx')}
      for (int j = 0; j < ${this.maxRank}; j++) {
        if (j >= sparseDims) {
          break;
        }
        oldSparseIx[j] = int(getValueAtPos(oldNnzIx * sparseDims + j, widthA, heightA, A));
      }

      int oldSparsePos = indexToPos(oldSparseIx, oldSparseStrides);
      int newSparsePos = oldSparsePos * nnzFraction + residualNNZ;
      int newSparseIx[${this.maxRank}];
      ${this.initIndex('newSparseIx')}
      ${this.posToIndex('newSparseStrides', 'newSparseIx', 'newSparsePos')}

      int ax = int(index[1]);

      for (int j = 0; j < ${this.maxRank}; j++) {
        if (j == ax) {
          res = float(newSparseIx[j]);
          break;
        }
      }

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['A'];
  }

  calc(input: ReshapeIndicesInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {A: input.A});
    }

    const outputShape = this.getOutputShape(input);

    const info = this.getCompilationInfo(input);

    return this.compute(
      outputShape,
      {A: input.A},
      {
        nnzFraction: info.nnzFraction,
        sparseDims: info.sparseDims,
        oldSparseStrides: this.pad(info.oldSparseStrides as number[]),
        newSparseStrides: this.pad(info.newSparseStrides as number[]),
      }
    );
  }

  getOutputShape(input: ReshapeIndicesInput): readonly number[] {
    const oldSparseSize = getSize(input.sparseShape);

    const sparseShape = [];
    let sparseSize = 1;
    for (let i = 0; i < input.shape.length; i++) {
      if (sparseSize < oldSparseSize) {
        sparseSize *= input.shape[i];
        sparseShape.push(input.shape[i]);
      } else {
        break;
      }
    }

    const nnzFraction = sparseSize / oldSparseSize;
    const nnz = input.nnz * nnzFraction;

    return [nnz, sparseShape.length];
  }

  compile(info: ReshapeIndicesInfo) {
    super.compile(info);
  }

  getCompilationInfo(input: ReshapeIndicesInput): ReshapeIndicesInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    const oldSparseSize = getSize(input.sparseShape);

    const sparseShape = [];
    const denseShape = [];
    let sparseSize = 1;
    for (let i = 0; i < input.shape.length; i++) {
      if (sparseSize < oldSparseSize) {
        sparseSize *= input.shape[i];
        sparseShape.push(input.shape[i]);
      } else {
        denseShape.push(input.shape[i]);
      }
    }

    const oldSparseStrides = computeStrides(input.sparseShape);
    const newSparseStrides = computeStrides(sparseShape);

    const nnzFraction = sparseSize / oldSparseSize;

    return {
      shapeA: input.A.shape,
      widthA: input.A.memory.width,
      heightA: input.A.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      sparseDims: input.sparseShape.length,
      newSparseStrides,
      oldSparseStrides,
      nnzFraction,
    };
  }

  getInputInfoString(input: ReshapeIndicesInput): string {
    return `${input.A.shape}-${input.nnz}-${input.shape}-${input.sparseShape}`;
  }
}

export const defaultReshapeIndicesD = new Dispatcher(
  (dtype: DTypeGpu) => new ReshapeIndicesOperation(gpuConstructor, dtype)
);

export function reshapeGPU<DTpe extends DTypeGpu>(
  tensor: SparseTensor<DTpe>,
  values: GPUTensor<DTpe>,
  indices: GPUTensor<'uint32'>,
  shape: readonly number[],
  copy: boolean
): SparseTensor<DTpe> {
  const oldSparseSize = getSize(tensor.getSparseShape());

  const sparseShape = [];
  const denseShape = [];
  let sparseSize = 1;
  for (let i = 0; i < shape.length; i++) {
    if (sparseSize < oldSparseSize) {
      sparseSize *= shape[i];
      sparseShape.push(shape[i]);
    } else {
      denseShape.push(shape[i]);
    }
  }

  const nnzFraction = sparseSize / oldSparseSize;
  const nnz = tensor.nnz * nnzFraction;

  const newValues = values.reshape([nnz, ...denseShape], copy);
  let newIndices: GPUTensor<'uint32'>;
  if (!copy && compareShapes(sparseShape, tensor.getSparseShape())) {
    newIndices = indices;
  } else {
    newIndices = defaultReshapeIndicesD.calc(
      {
        A: indices,
        sparseShape: tensor.getSparseShape(),
        shape: shape,
        nnz: tensor.nnz,
      },
      'uint32'
    ) as GPUTensor<'uint32'>;
  }

  return new SparseTensor(newValues, newIndices, shape, denseShape.length);
}
