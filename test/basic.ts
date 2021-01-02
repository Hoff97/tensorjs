import Tensor from '../js/types';

const epsilon = 0.00001;

// eslint-disable-next-line no-unused-vars
type TensorConstructor = (shape: ReadonlyArray<number>, values: number[]) => Tensor

export default function testBasic(name: string, constructor: TensorConstructor, wait?: Promise<void>) {
  describe(`${name} exp`, () => {
    it('should compute the exponent', async () => {
      if (wait) {
        await wait;
      }

      const input = constructor([2, 2], [-1, 0, 1, 2]);
      const expected = constructor([2, 2], [0.367879441, 1, 2.718281828, 7.389056099]);

      expect(await input.exp().compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} log`, () => {
    it('should compute the log', async () => {
      if (wait) {
        await wait;
      }

      const input = constructor([2, 2], [0.367879441, 1, 2.718281828, 7.389056099]);
      const expected = constructor([2, 2], [-1, 0, 1, 2]);

      expect(await input.log().compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} sqrt`, () => {
    it('should compute the sqrt', async () => {
      const input = constructor([2, 2], [1, 4, 9, 16]);
      const expected = constructor([2, 2], [1, 2, 3, 4]);

      expect(await input.sqrt().compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} add`, () => {
    it('should compute the pointwise addition', async () => {
      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([2, 2], [2, 6, 12, 20]);

      expect(await a.add(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} subtract`, () => {
    it('should compute the pointwise subtraction', async () => {
      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([2, 2], [0, 2, 6, 12]);

      expect(await a.subtract(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} divide`, () => {
    it('should compute the pointwise division', async () => {
      const a = constructor([2, 3], [1, 4, 9, 16, 21, 28]);
      const b = constructor([2, 3], [1, 2, 3, 4, 7, 7]);
      const expected = constructor([2, 3], [1, 2, 3, 4, 3, 4]);

      expect(await a.divide(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} multiply`, () => {
    it('should compute the pointwise division', async () => {
      const a = constructor([2, 2], [1, 2, 3, 4]);
      const b = constructor([2, 2], [5, 6, 7, 8]);
      const expected = constructor([2, 2], [5, 12, 21, 32]);

      expect(await a.multiply(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} matMul`, () => {
    it('should compute the matrix product', async () => {
      const a = constructor([2, 2], [1, 2, 3, 4]);
      const b = constructor([2, 2], [5, 6, 7, 8]);
      const expected = constructor([2, 2], [19, 22, 43, 50]);

      expect(await a.matMul(b).compare(expected, epsilon)).toBeTruthy();
    });

    it('should be the dot product for row/column vectors', async () => {
      const a = constructor([1, 3], [1, 2, 3]);
      const b = constructor([3, 1], [4, 5, 6]);
      const expected = constructor([1, 1], [32]);

      expect(await a.matMul(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  fdescribe(`${name} sum`, () => {
    it('should compute the sum of the whole matrix without axes', async () => {
      const a = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([1], [10]);

      console.log('Result', await a.sum().getValues());

      expect(await a.sum().compare(expected, epsilon)).toBeTruthy();
    });

    it('should compute the column wise sum with axes=0', async () => {
      const a = constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const expected = constructor([3], [5, 7, 9]); //yay

      expect(await a.sum(0).compare(expected, epsilon)).toBeTruthy();
    });

    it('should compute the row wise sum with axes=1', async () => {
      const a = constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const expected = constructor([2], [6, 15]);

      expect(await a.sum(1).compare(expected, epsilon)).toBeTruthy();
    });

    it('should work with multiple summation axes', async () => {
      const a = constructor([2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);
      const expected1 = constructor([2], [78, 222]);
      const expected2 = constructor([3], [68, 100, 132]);
      const expected3 = constructor([4], [66, 72, 78, 84]);

      expect(await a.sum([1,2]).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.sum([0,2]).compare(expected2, epsilon)).toBeTruthy();
      expect(await a.sum([0,1]).compare(expected3, epsilon)).toBeTruthy();
    });
  });
}
