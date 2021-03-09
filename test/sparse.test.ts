import {CPUTensor} from '../lib/tensor/cpu/tensor';
import {SparseTensor} from '../lib/tensor/sparse/tensor';

describe('Create sparse tensor', () => {
  it('should get the same values back', async () => {
    const nnz = 4;
    const shape = [3, 3];

    const indiceVals = [0, 0, 1, 1, 2, 1, 2, 2];
    const indiceTensor = new CPUTensor(
      [nnz, shape.length],
      indiceVals,
      'uint32'
    );

    const valueVals = [1, 2, 3, 4];
    const valueTensor = new CPUTensor([nnz], valueVals, 'float32');

    const tensor = new SparseTensor(valueTensor, indiceTensor, shape);

    const allVals = [1, 0, 0, 0, 2, 0, 0, 3, 4];

    expect(tensor.denseDims).toEqual(0);
    expect(tensor.nnz).toEqual(nnz);

    expect(tensor.getSparseShape()).toEqual([3, 3]);
    expect(tensor.getDenseShape()).toEqual([]);
    expect(await tensor.getValues()).toEqual(new Float32Array(allVals));
  });

  it('should work with nonzero dense dimensions', async () => {
    const nnz = 4;
    const shape = [3, 3, 2];
    const denseDims = 1;

    const indiceVals = [0, 0, 1, 1, 2, 1, 2, 2];
    const indiceTensor = new CPUTensor(
      [nnz, shape.length - denseDims],
      indiceVals,
      'uint32'
    );

    const valueVals = [1, 2, 3, 4, 5, 6, 7, 8];
    const valueTensor = new CPUTensor([nnz], valueVals, 'float32');

    const tensor = new SparseTensor(
      valueTensor,
      indiceTensor,
      shape,
      denseDims
    );

    const allVals = [
      1,
      2,
      0,
      0,
      0,
      0,
      /*Row 2*/ 0,
      0,
      3,
      4,
      0,
      0,
      /*Row 3*/ 0,
      0,
      5,
      6,
      7,
      8,
    ];

    expect(tensor.denseDims).toEqual(1);
    expect(tensor.nnz).toEqual(nnz);

    expect(tensor.getSparseShape()).toEqual([3, 3]);
    expect(tensor.getDenseShape()).toEqual([2]);
    expect(await tensor.getValues()).toEqual(new Float32Array(allVals));
  });

  it('should work with reshaping', async () => {
    const nnz = 1;
    const shape = [2, 2];
    const denseDims = 1;

    const indiceVals = [0];
    const indiceTensor = new CPUTensor(
      [nnz, shape.length - denseDims],
      indiceVals,
      'uint32'
    );

    const valueVals = [1, 2];
    const valueTensor = new CPUTensor(
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
    const nnz = 3;
    const shape = [3, 2, 2];
    const denseDims = 1;

    const indiceVals = [0, 0, 1, 1, 2, 1];
    const indiceTensor = new CPUTensor(
      [nnz, shape.length - denseDims],
      indiceVals,
      'uint32'
    );

    const valueVals = [1, 2, 3, 4, 5, 6];
    const valueTensor = new CPUTensor(
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
    const nnz = 2;
    const shape = [3, 2, 2];
    const denseDims = 2;

    const indiceVals = [0, 2];
    const indiceTensor = new CPUTensor(
      [nnz, shape.length - denseDims],
      indiceVals,
      'uint32'
    );

    const valueVals = [1, 2, 3, 4, 5, 6, 7, 8];
    const valueTensor = new CPUTensor(
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
    const nnz = 3;
    const shape = [2, 3, 2, 2];
    const denseDims = 2;

    const indiceVals = [0, 0, 0, 2, 1, 1];
    const indiceTensor = new CPUTensor(
      [nnz, shape.length - denseDims],
      indiceVals,
      'uint32'
    );

    const valueVals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const valueTensor = new CPUTensor(
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
});
