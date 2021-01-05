import Tensor from '../js/types';

const DELTA = 0.00001;

// eslint-disable-next-line no-unused-vars
type TensorConstructor = (shape: ReadonlyArray<number>, values: number[]) => Tensor

export default function testPool(name: string, constructor: TensorConstructor, wait?: Promise<void>) {
  describe(`${name} sum`, () => {
    it('should compute the sum of the whole matrix without axes', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([1], [10]);

      expect(await a.sum().compare(expected, DELTA)).toBeTruthy();
    });

    it('should compute the column wise sum with axes=0', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const expected1 = constructor([3], [5, 7, 9]);
      const expected2 = constructor([1, 3], [5, 7, 9]);

      expect(await a.sum(0).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.sum(0, true).compare(expected2, DELTA)).toBeTruthy();
    });

    it('should compute the row wise sum with axes=1', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const expected1 = constructor([2], [6, 15]);
      const expected2 = constructor([2, 1], [6, 15]);

      expect(await a.sum(1).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.sum(1, true).compare(expected2, DELTA)).toBeTruthy();
    });

    it('should work with multiple summation axes', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);
      const expected1 = constructor([2], [78, 222]);
      const expected11 = constructor([2, 1, 1], [78, 222]);
      const expected2 = constructor([3], [68, 100, 132]);
      const expected21 = constructor([1,3,1], [68, 100, 132]);
      const expected3 = constructor([4], [66, 72, 78, 84]);
      const expected31 = constructor([1,1,4], [66, 72, 78, 84]);

      expect(await a.sum([1,2]).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.sum([1,2], true).compare(expected11, DELTA)).toBeTruthy();
      expect(await a.sum([0,2]).compare(expected2, DELTA)).toBeTruthy();
      expect(await a.sum([0,2], true).compare(expected21, DELTA)).toBeTruthy();
      expect(await a.sum([0,1]).compare(expected3, DELTA)).toBeTruthy();
      expect(await a.sum([0,1], true).compare(expected31, DELTA)).toBeTruthy();
    });
  });

  describe(`${name} product`, () => {
    it('should compute the product of the whole matrix without axes', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([1], [24]);

      expect(await a.product().compare(expected, DELTA)).toBeTruthy();
    });

    it('should compute the column wise product with axes=0', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const expected1 = constructor([3], [4, 10, 18]);
      const expected2 = constructor([1, 3], [4, 10, 18]);

      expect(await a.product(0).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.product(0, true).compare(expected2, DELTA)).toBeTruthy();
    });

    it('should compute the row wise product with axes=1', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const expected1 = constructor([2], [6, 120]);
      const expected2 = constructor([2, 1], [6, 120]);

      expect(await a.product(1).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.product(1, true).compare(expected2, DELTA)).toBeTruthy();
    });

    it('should work with multiple product axes', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 1.5, 2.5, 0.5, 1.5, 3.5, 4.5]);
      const expected1 = constructor([3], [4.0320000648498535, 12600, 0.02835000306367874]);
      const expected2 = constructor([1,3,1], [4.0320000648498535, 12600, 0.02835000306367874]);

      expect(await a.product([0,2]).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.product([0,2], true).compare(expected2, DELTA)).toBeTruthy();
    });
  });

  describe(`${name} max`, () => {
    it('should compute the max of the whole matrix without axes', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([1], [4]);

      expect(await a.max().compare(expected, DELTA)).toBeTruthy();
    });

    it('should compute the column wise max with axes=0', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 5, 3, 4, 2, 6]);
      const expected1 = constructor([3], [4, 5, 6]);
      const expected2 = constructor([1,3], [4, 5, 6]);

      expect(await a.max(0).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.max(0, true).compare(expected2, DELTA)).toBeTruthy();
    });

    it('should compute the row wise max with axes=1', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 3, 2, 6, 4, 5]);
      const expected1 = constructor([2], [3, 6]);
      const expected2 = constructor([2,1], [3, 6]);

      expect(await a.max(1).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.max(1, true).compare(expected2, DELTA)).toBeTruthy();
    });

    it('should work with multiple max axes', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);
      const expected1 = constructor([2], [12, 24]);
      const expected11 = constructor([2, 1, 1], [12, 24]);
      const expected2 = constructor([3], [16, 20, 24]);
      const expected21 = constructor([1,3,1], [16, 20, 24]);
      const expected3 = constructor([4], [21, 22, 23, 24]);
      const expected31 = constructor([1,1,4], [21, 22, 23, 24]);

      expect(await a.max([1,2]).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.max([1,2], true).compare(expected11, DELTA)).toBeTruthy();
      expect(await a.max([0,2]).compare(expected2, DELTA)).toBeTruthy();
      expect(await a.max([0,2], true).compare(expected21, DELTA)).toBeTruthy();
      expect(await a.max([0,1]).compare(expected3, DELTA)).toBeTruthy();
      expect(await a.max([0,1], true).compare(expected31, DELTA)).toBeTruthy();
    });
  });

  describe(`${name} min`, () => {
    it('should compute the min of the whole matrix without axes', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([1], [1]);

      expect(await a.min().compare(expected, DELTA)).toBeTruthy();
    });

    it('should compute the column wise min with axes=0', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 5, 3, 4, 2, 6]);
      const expected1 = constructor([3], [1, 2, 3]);
      const expected2 = constructor([1, 3], [1, 2, 3]);

      expect(await a.min(0).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.min(0, true).compare(expected2, DELTA)).toBeTruthy();
    });

    it('should compute the row wise min with axes=1', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 3, 2, 6, 4, 5]);
      const expected1 = constructor([2], [1, 4]);
      const expected2 = constructor([2,1], [1, 4]);

      expect(await a.min(1).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.min(1, true).compare(expected2, DELTA)).toBeTruthy();
    });

    it('should work with multiple min axes', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);
      const expected1 = constructor([2], [1, 13]);
      const expected11 = constructor([2,1,1], [1, 13]);
      const expected2 = constructor([3], [1, 5, 9]);
      const expected21 = constructor([1,3,1], [1, 5, 9]);
      const expected3 = constructor([4], [1, 2, 3, 4]);
      const expected31 = constructor([1,1,4], [1, 2, 3, 4]);

      expect(await a.min([1,2]).compare(expected1, DELTA)).toBeTruthy();
      expect(await a.min([0,2]).compare(expected2, DELTA)).toBeTruthy();
      expect(await a.min([0,1]).compare(expected3, DELTA)).toBeTruthy();
      
      expect(await a.min([1,2], true).compare(expected11, DELTA)).toBeTruthy();
      expect(await a.min([0,2], true).compare(expected21, DELTA)).toBeTruthy();
      expect(await a.min([0,1], true).compare(expected31, DELTA)).toBeTruthy();
    });
  });
}
