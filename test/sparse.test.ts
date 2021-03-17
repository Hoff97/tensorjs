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
  });
}
