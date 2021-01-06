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

      input.delete();
      expected.delete();
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

      input.delete();
      expected.delete();
    });
  });

  describe(`${name} sqrt`, () => {
    it('should compute the sqrt', async () => {
      if (wait) {
        await wait;
      }

      const input = constructor([2, 2], [1, 4, 9, 16]);
      const expected = constructor([2, 2], [1, 2, 3, 4]);

      expect(await input.sqrt().compare(expected, epsilon)).toBeTruthy();

      input.delete();
      expected.delete();
    });
  });

  describe(`${name} add`, () => {
    it('should compute the pointwise addition', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([2, 2], [2, 6, 12, 20]);

      expect(await a.add(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('Should work with broadcasting', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([1], [1]);
      const c = constructor([2], [1, 2]);
      const expected1 = constructor([2, 2], [2, 5, 10, 17]);
      const expected2 = constructor([2, 2], [2, 6, 10, 18]);

      expect(await a.add(b).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.add(c).compare(expected2, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      c.delete();
      expected1.delete();
      expected2.delete();
    });

    it('Should work with broadcasting with both tensors', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([1, 3, 2], [1,2,3,4,5,6]);
      const b = constructor([2,3,1,1], [1,2,3,4,5,6]);
      const expected1 = constructor([2,3,3,2], [2,3,4,5,6,7,3,4,5,6,7,8,4,5,6,7,8,9,5,6,7,8,9,10,6,7,8,9,10,11,7,8,9,10,11,12]);

      expect(await a.add(b).compare(expected1, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected1.delete();
    });
  });

  describe(`${name} subtract`, () => {
    it('should compute the pointwise subtraction', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([2, 2], [1, 2, 3, 4]);
      const expected = constructor([2, 2], [0, 2, 6, 12]);

      expect(await a.subtract(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('Should work with broadcasting', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([1], [1]);
      const c = constructor([2], [1, 2]);
      const expected1 = constructor([2, 2], [0, 3, 8, 15]);
      const expected2 = constructor([2, 2], [0, 2, 8, 14]);

      expect(await a.subtract(b).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.subtract(c).compare(expected2, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      c.delete();
      expected1.delete();
      expected2.delete();
    });
  });

  describe(`${name} divide`, () => {
    it('should compute the pointwise division', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 4, 9, 16, 21, 28]);
      const b = constructor([2, 3], [1, 2, 3, 4, 7, 7]);
      const expected = constructor([2, 3], [1, 2, 3, 4, 3, 4]);

      expect(await a.divide(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('Should work with broadcasting', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([1], [1]);
      const c = constructor([2], [1, 2]);
      const expected1 = constructor([2, 2], [1, 4, 9, 16]);
      const expected2 = constructor([2, 2], [1, 2, 9, 8]);

      expect(await a.divide(b).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.divide(c).compare(expected2, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      c.delete();
      expected1.delete();
      expected2.delete();
    });
  });

  describe(`${name} multiply`, () => {
    it('should compute the pointwise division', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const b = constructor([2, 2], [5, 6, 7, 8]);
      const expected = constructor([2, 2], [5, 12, 21, 32]);

      expect(await a.multiply(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('Should work with broadcasting', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 4, 9, 16]);
      const b = constructor([1], [1]);
      const c = constructor([2], [1, 2]);
      const expected1 = constructor([2, 2], [1, 4, 9, 16]);
      const expected2 = constructor([2, 2], [1, 8, 9, 32]);

      expect(await a.multiply(b).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.multiply(c).compare(expected2, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      c.delete();
      expected1.delete();
      expected2.delete();
    });
  });

  describe(`${name} matMul`, () => {
    it('should compute the matrix product', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const b = constructor([2, 2], [5, 6, 7, 8]);
      const expected = constructor([2, 2], [19, 22, 43, 50]);

      expect(await a.matMul(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should be the dot product for row/column vectors', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([1, 3], [1, 2, 3]);
      const b = constructor([3, 1], [4, 5, 6]);
      const expected = constructor([1, 1], [32]);

      expect(await a.matMul(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });
  });

  describe(`${name} concat`, () => {
    it('should work on all axis', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3,4], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]);

      const b = constructor([1,3,4], [1,2,3,4,5,6,7,8,9,10,11,12]);
      const c = constructor([2,1,4], [1,2,3,4,5,6,7,8]);
      const d = constructor([2,3,1], [1,2,3,4,5,6]);

      const expected1 = constructor([3,3,4], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24, 1,2,3,4,5,6,7,8,9,10,11,12]);
      const expected2 = constructor([2,4,4], [1,2,3,4,5,6,7,8,9,10,11,12, 1,2,3,4, 13,14,15,16,17,18,19,20,21,22,23,24, 5,6,7,8]);
      const expected3 = constructor([2,3,5], [1,2,3,4, 1, 5,6,7,8, 2, 9,10,11,12, 3, 13,14,15,16, 4, 17,18,19,20, 5, 21,22,23,24, 6]);

      expect(await a.concat(b, 0).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.concat(c, 1).compare(expected2, epsilon)).toBeTruthy();
      expect(await a.concat(d, 2).compare(expected3, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      c.delete();
      d.delete();
      expected1.delete();
      expected2.delete();
      expected3.delete();
    });
  });

  describe(`${name} softmax`, () => {
    it('should work on all axis', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3], [1,2,3,4,5,6]);

      const expected1 = constructor([2,3], [0.0474258736, 0.0474258736, 0.0474258736, 0.9525741339, 0.9525741339, 0.9525741339]);
      const expected2 = constructor([2,3], [0.0900305733, 0.2447284758, 0.6652409434, 0.0900305733, 0.2447284758, 0.6652409434]);

      expect(await a.softmax(0).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.softmax(1).compare(expected2, epsilon)).toBeTruthy();

      a.delete();
      expected1.delete();
      expected2.delete();
    });
  });
}
