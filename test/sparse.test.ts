import {CPUTensor} from '../lib/tensor/cpu/tensor';
import {GPUTensor} from '../lib/tensor/gpu/tensor';
import {SparseTensor} from '../lib/tensor/sparse/tensor';
import {wasmLoaded, WASMTensor} from '../lib/tensor/wasm/tensor';
import Tensor, {DType} from '../lib/types';
import {toCPU, toGPU, toWASM} from '../lib/util/convert';

interface Backend {
  name: string;
  constructor: <DTpe extends DType>(
    shape: ReadonlyArray<number>,
    values: number[],
    dtype: DTpe
  ) => Tensor<DTpe>;
  toBackend: <DTpe extends DType>(
    tensor: Tensor<DTpe>
  ) => Promise<Tensor<DTpe>>;
  wait?: Promise<void>;
}

const epsilon = 0.0001;

const backends: Backend[] = [
  {
    name: 'CPU',
    constructor: <DTpe extends DType>(
      shape: ReadonlyArray<number>,
      values: number[],
      dtype: DTpe
    ) => new CPUTensor(shape, values, dtype),
    toBackend: <DTpe extends DType>(tensor: Tensor<DTpe>) => toCPU(tensor),
  },
  {
    name: 'WASM',
    constructor: <DTpe extends DType>(
      shape: ReadonlyArray<number>,
      values: number[],
      dtype: DTpe
    ) => new WASMTensor(values, new Uint32Array(shape), dtype as any),
    toBackend: <DTpe extends DType>(tensor: Tensor<DTpe>) => toWASM(tensor),
    wait: wasmLoaded,
  } /*,
  {
    name: 'GPU',
    constructor: <DTpe extends DType>(
      shape: ReadonlyArray<number>,
      values: number[],
      dtype: DTpe
    ) => new GPUTensor(values, shape, dtype as any),
    toBackend: <DTpe extends DType>(tensor: Tensor<DTpe>) => toGPU(tensor),
  }*/,
];

for (const backend of backends) {
  describe(`Sparse tensor on ${backend.name}`, () => {
    it('should get the same values back after creation', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3];

      const indiceVals = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensor = backend.constructor(
        [nnz, shape.length],
        indiceVals,
        'uint32'
      );

      const valueVals = [1, 2, 3, 4];
      const valueTensor = backend.constructor([nnz], valueVals, 'float32');

      const tensor = new SparseTensor(valueTensor, indiceTensor, shape);

      const allVals = [1, 0, 0, 0, 2, 0, 0, 3, 4];

      expect(tensor.denseDims).toEqual(0);
      expect(tensor.nnz).toEqual(nnz);

      expect(tensor.getSparseShape()).toEqual([3, 3]);
      expect(tensor.getDenseShape()).toEqual([]);
      expect(await tensor.getValues()).toEqual(new Float32Array(allVals));
    });

    it('should work with creation from dense tensor', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const allVals = [1, 0, 0, 0, 2, 0, 0, 3, 4];
      const input = new CPUTensor([3, 3], allVals);
      const fromDense: SparseTensor = (await backend.toBackend(
        SparseTensor.fromDense(input)
      )) as SparseTensor;

      const nnz = 4;
      const shape = [3, 3];

      const indiceVals = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensor = backend.constructor(
        [nnz, shape.length],
        indiceVals,
        'uint32'
      );

      const valueVals = [1, 2, 3, 4];
      const valueTensor = backend.constructor([nnz], valueVals, 'float32');
      const expected = new SparseTensor(valueTensor, indiceTensor, shape);

      expect(fromDense.nnz).toBe(4);
      expect(fromDense.denseDims).toBe(0);
      expect(fromDense.shape).toEqual([3, 3]);

      expect(await fromDense.compare(expected)).toBeTrue();
      expect(await fromDense.values.compare(expected.values)).toBeTrue();
      expect(await fromDense.indices.compare(expected.indices)).toBeTrue();
    });

    it('should work with nonzero dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceVals = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensor = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceVals,
        'uint32'
      );

      const valueVals = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensor = backend.constructor([nnz * 2], valueVals, 'float32');

      const tensor = new SparseTensor(
        valueTensor,
        indiceTensor,
        shape,
        denseDims
      );

      const allVals = [1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8];

      expect(tensor.denseDims).toEqual(1);
      expect(tensor.nnz).toEqual(nnz);

      expect(tensor.getSparseShape()).toEqual([3, 3]);
      expect(tensor.getDenseShape()).toEqual([2]);
      expect(await tensor.getValues()).toEqual(new Float32Array(allVals));
    });

    it('should work with reshaping', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 1;
      const shape = [2, 2];
      const denseDims = 1;

      const indiceVals = [0];
      const indiceTensor = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceVals,
        'uint32'
      );

      const valueVals = [1, 2];
      const valueTensor = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueVals,
        'float32'
      );

      const tensor = new SparseTensor(
        valueTensor,
        indiceTensor,
        shape,
        denseDims
      );

      const result = tensor.reshape([4]) as SparseTensor;

      expect(result.denseDims).toEqual(0);
      expect(result.nnz).toEqual(2);

      expect(result.getSparseShape()).toEqual([4]);
      expect(result.getDenseShape()).toEqual([]);

      expect(await result.getValues()).toEqual(await tensor.getValues());
    });

    it('should work with reshaping only across sparse dims', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 3;
      const shape = [3, 2, 2];
      const denseDims = 1;

      const indiceVals = [0, 0, 1, 1, 2, 1];
      const indiceTensor = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceVals,
        'uint32'
      );

      const valueVals = [1, 2, 3, 4, 5, 6];
      const valueTensor = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueVals,
        'float32'
      );

      const tensor = new SparseTensor(
        valueTensor,
        indiceTensor,
        shape,
        denseDims
      );

      const result = tensor.reshape([6, 2]) as SparseTensor;

      expect(result.denseDims).toEqual(1);
      expect(result.nnz).toEqual(3);

      expect(result.getSparseShape()).toEqual([6]);
      expect(result.getDenseShape()).toEqual([2]);

      expect(await result.getValues()).toEqual(await tensor.getValues());
    });

    it('should work with reshaping only across dense dims', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 2;
      const shape = [3, 2, 2];
      const denseDims = 2;

      const indiceVals = [0, 2];
      const indiceTensor = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceVals,
        'uint32'
      );

      const valueVals = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensor = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueVals,
        'float32'
      );

      const tensor = new SparseTensor(
        valueTensor,
        indiceTensor,
        shape,
        denseDims
      );

      const result = tensor.reshape([3, 4]) as SparseTensor;

      expect(result.denseDims).toEqual(1);
      expect(result.nnz).toEqual(2);

      expect(result.getSparseShape()).toEqual([3]);
      expect(result.getDenseShape()).toEqual([4]);

      expect(await result.getValues()).toEqual(await tensor.getValues());
    });

    it('should work with reshaping across dense and sparse dims', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 3;
      const shape = [2, 3, 2, 2];
      const denseDims = 2;

      const indiceVals = [0, 0, 0, 2, 1, 1];
      const indiceTensor = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceVals,
        'uint32'
      );

      const valueVals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
      const valueTensor = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueVals,
        'float32'
      );

      const tensor = new SparseTensor(
        valueTensor,
        indiceTensor,
        shape,
        denseDims
      );

      const result = tensor.reshape([12, 2]) as SparseTensor;

      expect(result.denseDims).toEqual(1);
      expect(result.nnz).toEqual(6);

      expect(result.getSparseShape()).toEqual([12]);
      expect(result.getDenseShape()).toEqual([2]);

      expect(await result.getValues()).toEqual(await tensor.getValues());
    });

    it('should work with concatenating along sparse axis', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnzA = 2;
      const nnzB = 2;
      const shape = [2, 2];

      const indiceValsA = [0, 0, 1, 1];
      const indiceTensorA = backend.constructor(
        [nnzA, shape.length],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2];
      const valueTensorA = backend.constructor([nnzA], valueValsA, 'float32');
      const tensorA = new SparseTensor(valueTensorA, indiceTensorA, shape);

      const indiceValsB = [0, 0, 1, 0];
      const indiceTensorB = backend.constructor(
        [nnzB, shape.length],
        indiceValsB,
        'uint32'
      );
      const valueValsB = [3, 4];
      const valueTensorB = backend.constructor([nnzB], valueValsB, 'float32');
      const tensorB = new SparseTensor(valueTensorB, indiceTensorB, shape);

      const indiceValsResult1 = [0, 0, 1, 1, 2, 0, 3, 0];
      const indiceTensorResult1 = backend.constructor(
        [nnzA + nnzB, shape.length],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [1, 2, 3, 4];
      const valueTensorResult1 = backend.constructor(
        [nnzA + nnzB],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [4, 2]
      );

      const res1 = tensorA.concat(tensorB, 0) as SparseTensor;

      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([4, 2]);

      expect(await res1.compare(tensorResult1)).toBeTrue();
    });

    it('should work with repeating along sparse and dense axis', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 2;
      const shape = [3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 2, 3, 5];
      const indiceTensorResult1 = backend.constructor(
        [4, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4];
      const valueTensorResult1 = backend.constructor(
        [4, 4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [6, 4],
        denseDims
      );

      const res1 = tensorA.repeat([2, 2]) as SparseTensor;

      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([6, 4]);

      expect(await res1.compare(tensorResult1)).toBeTrue();
    });

    it('should work with repeating along sparse axis', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnzA = 2;
      const shape = [2, 2];

      const indiceValsA = [0, 0, 1, 1];
      const indiceTensorA = backend.constructor(
        [nnzA, shape.length],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2];
      const valueTensorA = backend.constructor([nnzA], valueValsA, 'float32');
      const tensorA = new SparseTensor(valueTensorA, indiceTensorA, shape);

      const indiceValsResult1 = [
        0,
        0,
        1,
        1,
        0,
        2,
        1,
        3,
        0,
        4,
        1,
        5,
        2,
        0,
        3,
        1,
        2,
        2,
        3,
        3,
        2,
        4,
        3,
        5,
      ];
      const indiceTensorResult1 = backend.constructor(
        [12, shape.length],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
      const valueTensorResult1 = backend.constructor(
        [12],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [4, 6]
      );

      const res1 = tensorA.repeat([2, 3]) as SparseTensor;

      expect(res1.nnz).toBe(12);
      expect(res1.shape).toEqual([4, 6]);

      expect(await res1.compare(tensorResult1)).toBeTrue();
    });

    it('should work with sparse-dense matmul', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [1, 0, 0, 0, 2, 0, 0, 3, 4])
        )
      );

      const b = await backend.toBackend(
        new CPUTensor([3, 3], [5, 6, 7, 8, 9, 10, 11, 12, 13])
      );

      const expected = await backend.toBackend(
        new CPUTensor([3, 3], [5, 6, 7, 16, 18, 20, 68, 75, 82])
      );

      const result = a.matMul(b);

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with sparse-dense addition', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [1, 0, 0, 0, 2, 0, 0, 3, 4])
        )
      );

      const b = await backend.toBackend(new CPUTensor([2], [5, 6]));

      const result = a
        .reshape([3, 3, 1], false)
        .add(b.reshape([1, 1, 2], false));

      const expectedIx = [0, 0, 1, 1, 2, 1, 2, 2];
      const expectedIxTensor = backend.constructor(
        [4, 2],
        expectedIx,
        'uint32'
      );
      const expectedValues = [6, 7, 7, 8, 8, 9, 9, 10];
      const expectedVTensor = backend.constructor(
        [4, 2],
        expectedValues,
        'float32'
      );
      const expected = new SparseTensor(
        expectedVTensor,
        expectedIxTensor,
        [3, 3, 2],
        1
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with sparse-sparse addition', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [1, 0, 0, 0, 2, 0, 0, 3, 4])
        )
      );

      const b = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [5, 0, 0, 0, 6, 0, 0, 7, 8])
        )
      );

      const result = a.add(b);

      const expected = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [6, 0, 0, 0, 8, 0, 0, 10, 12])
        )
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with sparse-dense subtraction', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [1, 0, 0, 0, 2, 0, 0, 3, 4])
        )
      );

      const b = await backend.toBackend(new CPUTensor([2], [5, 6]));

      const result = a
        .reshape([3, 3, 1], false)
        .subtract(b.reshape([1, 1, 2], false));

      const expectedIx = [0, 0, 1, 1, 2, 1, 2, 2];
      const expectedIxTensor = backend.constructor(
        [4, 2],
        expectedIx,
        'uint32'
      );
      const expectedValues = [-4, -5, -3, -4, -2, -3, -1, -2];
      const expectedVTensor = backend.constructor(
        [4, 2],
        expectedValues,
        'float32'
      );
      const expected = new SparseTensor(
        expectedVTensor,
        expectedIxTensor,
        [3, 3, 2],
        1
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with sparse-sparse subtraction', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const b = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [1, 0, 0, 0, 2, 0, 0, 3, 4])
        )
      );

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [5, 0, 0, 0, 6, 0, 0, 7, 8])
        )
      );

      const result = a.subtract(b);

      const expected = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [4, 0, 0, 0, 4, 0, 0, 4, 4])
        )
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with sparse-dense multiplication', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [1, 0, 0, 0, 2, 0, 0, 3, 4])
        )
      );

      const b = await backend.toBackend(new CPUTensor([2], [5, 6]));

      const result = a
        .reshape([3, 3, 1], false)
        .multiply(b.reshape([1, 1, 2], false));

      const expectedIx = [0, 0, 1, 1, 2, 1, 2, 2];
      const expectedIxTensor = backend.constructor(
        [4, 2],
        expectedIx,
        'uint32'
      );
      const expectedValues = [5, 6, 10, 12, 15, 18, 20, 24];
      const expectedVTensor = backend.constructor(
        [4, 2],
        expectedValues,
        'float32'
      );
      const expected = new SparseTensor(
        expectedVTensor,
        expectedIxTensor,
        [3, 3, 2],
        1
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with sparse-sparse multiplication', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const b = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [1, 0, 0, 0, 2, 0, 0, 3, 4])
        )
      );

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [5, 0, 0, 0, 6, 0, 0, 7, 8])
        )
      );

      const result = a.multiply(b);

      const expected = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [5, 0, 0, 0, 12, 0, 0, 21, 32])
        )
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with sparse-dense division', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [7, 0, 0, 0, 8, 0, 0, 9, 10])
        )
      );

      const b = await backend.toBackend(new CPUTensor([2], [5, 6]));

      const result = a
        .reshape([3, 3, 1], false)
        .divide(b.reshape([1, 1, 2], false));

      const expectedIx = [0, 0, 1, 1, 2, 1, 2, 2];
      const expectedIxTensor = backend.constructor(
        [4, 2],
        expectedIx,
        'uint32'
      );
      const expectedValues = [
        7 / 5,
        7 / 6,
        8 / 5,
        8 / 6,
        9 / 5,
        9 / 6,
        10 / 5,
        10 / 6,
      ];
      const expectedVTensor = backend.constructor(
        [4, 2],
        expectedValues,
        'float32'
      );
      const expected = new SparseTensor(
        expectedVTensor,
        expectedIxTensor,
        [3, 3, 2],
        1
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with sparse-sparse division', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const b = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [1, 0, 0, 0, 2, 0, 0, 3, 4])
        )
      );

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [5, 0, 0, 0, 6, 0, 0, 7, 8])
        )
      );

      const result = a.divide(b);

      const expected = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 3], [5, 0, 0, 0, 3, 0, 0, 7 / 3, 2])
        )
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with summing over dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorResult1 = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [3, 7, 11, 15];
      const valueTensorResult1 = backend.constructor(
        [4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [3, 3]
      );

      const res1 = tensorA.sum(2) as SparseTensor;
      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([3, 3]);

      expect(await res1.compare(tensorResult1, epsilon)).toBeTrue();
    });

    it('should work with summing over sparse dimension 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.sum(0);

      const expected = await backend.constructor([4], [1, 7, 5, 2], 'float32');

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with summing over sparse dimension 2', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.sum(1);

      const expected = await backend.constructor([3], [3, 3, 9], 'float32');

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with summing squared over dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorResult1 = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [5, 25, 61, 113];
      const valueTensorResult1 = backend.constructor(
        [4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [3, 3]
      );

      const res1 = tensorA.sumSquare(2) as SparseTensor;
      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([3, 3]);

      expect(await res1.compare(tensorResult1, epsilon)).toBeTrue();
    });

    it('should work with summing squared over sparse dimension 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.sumSquare(0);

      const expected = await backend.constructor(
        [4],
        [1, 25, 25, 4],
        'float32'
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with summing squared over sparse dimension 2', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.sumSquare(1);

      const expected = await backend.constructor([3], [5, 9, 41], 'float32');

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with product over dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorResult1 = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [2, 12, 30, 56];
      const valueTensorResult1 = backend.constructor(
        [4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [3, 3]
      );

      const res1 = tensorA.product(2) as SparseTensor;
      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([3, 3]);

      expect(await res1.compare(tensorResult1, epsilon)).toBeTrue();
    });

    it('should work with product over sparse dimension 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.product(0);

      const expected = await backend.constructor([4], [1, 12, 5, 2], 'float32');

      expect(await result.compare(expected, epsilon)).toBeTrue();
    });

    it('should work with product over sparse dimension 2', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.product(1);

      const expected = await backend.constructor([3], [2, 3, 20], 'float32');

      expect(await result.compare(expected, epsilon)).toBeTrue();
    });

    it('should work with maximum over dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorResult1 = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [2, 4, 6, 8];
      const valueTensorResult1 = backend.constructor(
        [4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [3, 3]
      );

      const res1 = tensorA.max(2) as SparseTensor;
      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([3, 3]);

      expect(await res1.compare(tensorResult1, epsilon)).toBeTrue();
    });

    it('should work with maximum over sparse dimension 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.max(0);

      const expected = await backend.constructor([4], [1, 4, 5, 2], 'float32');

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with maximum over sparse dimension 2', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.max(1);

      const expected = await backend.constructor([3], [2, 3, 5], 'float32');

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with minimum over dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorResult1 = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [1, 3, 5, 7];
      const valueTensorResult1 = backend.constructor(
        [4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [3, 3]
      );

      const res1 = tensorA.min(2) as SparseTensor;
      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([3, 3]);

      expect(await res1.compare(tensorResult1, epsilon)).toBeTrue();
    });

    it('should work with minimum over sparse dimension 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.min(0);

      const expected = await backend.constructor([4], [1, 3, 5, 2], 'float32');

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with minimum over sparse dimension 2', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.min(1);

      const expected = await backend.constructor([3], [1, 3, 4], 'float32');

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with mean over dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorResult1 = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [3 / 2, 7 / 2, 11 / 2, 15 / 2];
      const valueTensorResult1 = backend.constructor(
        [4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [3, 3]
      );

      const res1 = tensorA.reduceMean(2) as SparseTensor;
      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([3, 3]);

      expect(await res1.compare(tensorResult1, epsilon)).toBeTrue();
    });

    it('should work with mean over sparse dimension 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.reduceMean(0);

      const expected = await backend.constructor(
        [4],
        [1, 7 / 2, 5, 2],
        'float32'
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with mean over sparse dimension 2', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.reduceMean(1);

      const expected = await backend.constructor(
        [3],
        [3 / 2, 3, 9 / 2],
        'float32'
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with mean squared over dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorResult1 = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [5 / 2, 25 / 2, 61 / 2, 113 / 2];
      const valueTensorResult1 = backend.constructor(
        [4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [3, 3]
      );

      const res1 = tensorA.reduceMeanSquare(2) as SparseTensor;
      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([3, 3]);

      expect(await res1.compare(tensorResult1, epsilon)).toBeTrue();
    });

    it('should work with mean squared over sparse dimension 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.reduceMeanSquare(0);

      const expected = await backend.constructor(
        [4],
        [1, 25 / 2, 25, 4],
        'float32'
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with mean squared over sparse dimension 2', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.reduceMeanSquare(1);

      const expected = await backend.constructor(
        [3],
        [5 / 2, 9, 41 / 2],
        'float32'
      );

      expect(await result.compare(expected)).toBeTrue();
    });

    it('should work with log sum over dense dimensions', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const nnz = 4;
      const shape = [3, 3, 2];
      const denseDims = 1;

      const indiceValsA = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorA = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsA,
        'uint32'
      );
      const valueValsA = [1, 2, 3, 4, 5, 6, 7, 8];
      const valueTensorA = backend.constructor(
        [nnz, ...shape.slice(shape.length - denseDims)],
        valueValsA,
        'float32'
      );
      const tensorA = new SparseTensor(
        valueTensorA,
        indiceTensorA,
        shape,
        denseDims
      );

      const indiceValsResult1 = [0, 0, 1, 1, 2, 1, 2, 2];
      const indiceTensorResult1 = backend.constructor(
        [nnz, shape.length - denseDims],
        indiceValsResult1,
        'uint32'
      );
      const valueValsResult1 = [
        Math.log(3),
        Math.log(7),
        Math.log(11),
        Math.log(15),
      ];
      const valueTensorResult1 = backend.constructor(
        [4],
        valueValsResult1,
        'float32'
      );
      const tensorResult1 = new SparseTensor(
        valueTensorResult1,
        indiceTensorResult1,
        [3, 3]
      );

      const res1 = tensorA.reduceLogSum(2) as SparseTensor;
      expect(res1.nnz).toBe(4);
      expect(res1.shape).toEqual([3, 3]);

      expect(await res1.compare(tensorResult1, epsilon)).toBeTrue();
    });

    it('should work with log sum over sparse dimension 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.reduceLogSum(0);

      const expected = await backend.constructor(
        [4],
        [Math.log(1), Math.log(7), Math.log(5), Math.log(2)],
        'float32'
      );

      expect(await result.compare(expected, epsilon)).toBeTrue();
    });

    it('should work with log sum over sparse dimension 2', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = await backend.toBackend(
        SparseTensor.fromDense(
          new CPUTensor([3, 4], [1, 0, 0, 2, 0, 3, 0, 0, 0, 4, 5, 0])
        )
      );

      const result = a.reduceLogSum(1);

      const expected = await backend.constructor(
        [3],
        [Math.log(3), Math.log(3), Math.log(9)],
        'float32'
      );

      expect(await result.compare(expected)).toBeTrue();
    });
  });
}
