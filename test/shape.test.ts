import {
  getSize, computeStrides, indexToPos, posToIndex
} from '../js/util/shape';

describe('Get size', () => {
  it('should return 0 on empty shape', () => {
    expect(getSize([])).toBe(0);
  });

  it('should return the first entry on shape of length 1', () => {
    expect(getSize([5])).toBe(5);
    expect(getSize([8])).toBe(8);
    expect(getSize([22])).toBe(22);
    expect(getSize([33])).toBe(33);
  });

  it('should compute the product of the shape', () => {
    expect(getSize([5, 10, 3])).toBe(5 * 10 * 3);
    expect(getSize([2, 33, 1])).toBe(2 * 33 * 1);
    expect(getSize([8, 7, 6])).toBe(8 * 7 * 6);
  });
});

describe('Compute strides', () => {
  it('should return an array of the same length', () => {
    expect(computeStrides([]).length).toBe(0);
    expect(computeStrides([1, 2, 3]).length).toBe(3);
    expect(computeStrides([1, 2, 3, 4, 5, 6]).length).toBe(6);
    expect(computeStrides([1, 2, 3, 4, 5, 6, 7, 8, 9]).length).toBe(9);
  });

  it('should work for rank 1', () => {
    expect(computeStrides([5])).toEqual([1]);
    expect(computeStrides([22])).toEqual([1]);
    expect(computeStrides([33])).toEqual([1]);
  });

  it('should ignore the first entry', () => {
    expect(computeStrides([5, 2, 3])).toEqual([6, 3, 1]);
    expect(computeStrides([22, 2, 3])).toEqual([6, 3, 1]);
    expect(computeStrides([31, 2, 3])).toEqual([6, 3, 1]);
  });
});

describe('Index to pos', () => {
  it('should return 0 for rank 0', () => {
    const shape: number[] = [];
    const strides = computeStrides(shape);

    expect(indexToPos([], strides)).toBe(0);
  });

  it('should return the index for rank 1', () => {
    const shape = [22];
    const strides = computeStrides(shape);

    expect(indexToPos([1], strides)).toBe(1);
    expect(indexToPos([5], strides)).toBe(5);
    expect(indexToPos([21], strides)).toBe(21);
  });

  it('should work for higher ranks', () => {
    const shape = [4, 3, 2];
    const strides = computeStrides(shape);

    expect(indexToPos([0, 1, 1], strides)).toBe(3);
    expect(indexToPos([1, 0, 1], strides)).toBe(7);
    expect(indexToPos([2, 2, 1], strides)).toBe(17);
    expect(indexToPos([3, 2, 1], strides)).toBe(23);
  });
});

describe('Pos to index', () => {
  it('should return empty vector for rank 0', () => {
    const shape: number[] = [];
    const strides = computeStrides(shape);

    expect(posToIndex(0, strides)).toEqual([]);
  });

  it('should return the index for rank 1', () => {
    const shape = [22];
    const strides = computeStrides(shape);

    expect(posToIndex(1, strides)).toEqual([1]);
    expect(posToIndex(5, strides)).toEqual([5]);
    expect(posToIndex(21, strides)).toEqual([21]);
  });

  it('should work for higher ranks', () => {
    const shape = [4, 3, 2];
    const strides = computeStrides(shape);

    expect(posToIndex(3, strides)).toEqual([0, 1, 1]);
    expect(posToIndex(7, strides)).toEqual([1, 0, 1]);
    expect(posToIndex(17, strides)).toEqual([2, 2, 1]);
    expect(posToIndex(23, strides)).toEqual([3, 2, 1]);
  });
});
