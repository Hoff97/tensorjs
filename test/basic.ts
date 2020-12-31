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

      expect(input.exp().compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} log`, () => {
    it('should compute the log', async () => {
      if (wait) {
        await wait;
      }

      const input = constructor([2, 2], [0.367879441, 1, 2.718281828, 7.389056099]);
      const expected = constructor([2, 2], [-1, 0, 1, 2]);

      expect(input.log().compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} sqrt`, () => {
    it('should compute the sqrt', () => {
      const input = constructor([2, 2], [1, 4, 9, 16]);
      const expected = constructor([2, 2], [1, 2, 3, 4]);

      expect(input.sqrt().compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} add`, () => {
    it('should compute the pointwise addition', () => {
      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([2, 2], [2, 6, 12, 20]);

      expect(a.add(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} subtract`, () => {
    it('should compute the pointwise subtraction', () => {
      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([2, 2], [0, 2, 6, 12]);

      expect(a.subtract(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} divide`, () => {
    it('should compute the pointwise division', () => {
      const a = constructor([2, 3], [1, 4, 9, 16, 21, 28]);
      const b = constructor([2, 3], [1, 2, 3, 4, 7, 7]);
      const expected = constructor([2, 3], [1, 2, 3, 4, 3, 4]);

      expect(a.divide(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} multiply`, () => {
    it('should compute the pointwise division', () => {
      const a = constructor([2, 2], [1, 2, 3, 4]);
      const b = constructor([2, 2], [5, 6, 7, 8]);
      const expected = constructor([2, 2], [5, 12, 21, 32]);

      expect(a.multiply(b).compare(expected, epsilon)).toBeTruthy();
    });
  });

  describe(`${name} matMul`, () => {
    it('should compute the matrix product', () => {
      const a = constructor([2, 2], [1, 2, 3, 4]);
      const b = constructor([2, 2], [5, 6, 7, 8]);
      const expected = constructor([2, 2], [19, 22, 43, 50]);

      expect(a.matMul(b).compare(expected, epsilon)).toBeTruthy();
    });

    it('should be the dot product for row/column vectors', () => {
      const a = constructor([1, 3], [1, 2, 3]);
      const b = constructor([3, 1], [4, 5, 6]);
      const expected = constructor([1, 1], [32]);

      expect(a.matMul(b).compare(expected, epsilon)).toBeTruthy();
    });
  });
}
