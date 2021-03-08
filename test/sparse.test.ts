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
});
