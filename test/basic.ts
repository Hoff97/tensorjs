/* eslint-disable prettier/prettier */
import Tensor from '../lib/types';

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

  describe(`${name} abs`, () => {
    it('should compute the absolute value', async () => {
      if (wait) {
        await wait;
      }

      const input = constructor([2, 3], [-5,1,0.01,-0.05,-100,1000]);
      const expected = constructor([2, 3], [5,1,0.01,0.05,100,1000]);

      expect(await input.abs().compare(expected, epsilon)).toBeTruthy();

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

  describe(`${name} power`, () => {
    it('should compute the pointwise power', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const b = constructor([2, 2], [2,3,2.5,3.5]);
      const expected = constructor([2, 2], [1, 8, 15.588457268, 128]);

      expect(await a.power(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('Should work with broadcasting', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1,2,3,4]);
      const b = constructor([1], [2]);
      const c = constructor([2], [2,3.5]);
      const expected1 = constructor([2, 2], [1,4,9,16]);
      const expected2 = constructor([2, 2], [1, 11.313708499, 9, 128]);

      expect(await a.power(b).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.power(c).compare(expected2, epsilon)).toBeTruthy();

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

  describe(`${name} gemm`, () => {
    it('should compute the matrix product', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const b = constructor([2, 2], [5, 6, 7, 8]);
      const expected = constructor([2, 2], [19, 22, 43, 50]);

      expect(await a.gemm(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should work with non-square matrices', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const b = constructor([3, 2], [7,8,9,10,11,12]);
      const expected = constructor([2, 2], [58, 64, 139, 154]);

      expect(await a.gemm(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should work with a transposed', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([3,2], [1,4,2,5,3,6]);
      const b = constructor([3, 2], [7,8,9,10,11,12]);
      const expected = constructor([2, 2], [58, 64, 139, 154]);

      expect(await a.gemm(b, true).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should work with a and b transposed', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([3,2], [1,4,2,5,3,6]);
      const b = constructor([2,3], [7,9,11,8,10,12]);
      const expected = constructor([2, 2], [58, 64, 139, 154]);

      expect(await a.gemm(b, true, true).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should work with b transposed', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3], [1,2,3,4,5,6]);
      const b = constructor([2,3], [7,9,11,8,10,12]);
      const expected = constructor([2, 2], [58, 64, 139, 154]);

      expect(await a.gemm(b, false, true).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should work with batches', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,2,3], [1,2,3,4,5,6,7,8,9,10,11,12]);
      const b = constructor([2,3,2], [7,8,9,10,11,12,13,14,15,16,17,18]);
      const expected = constructor([2, 2, 2], [58, 64, 139, 154, 364, 388, 499, 532]);

      expect(await a.gemm(b).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should work with batches and a transposed', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3,2], [1,4,2,5,3,6,7,10,8,11,9,12]);
      const b = constructor([2,3,2], [7,8,9,10,11,12,13,14,15,16,17,18]);
      const expected = constructor([2, 2, 2], [58, 64, 139, 154, 364, 388, 499, 532]);

      expect(await a.gemm(b, true).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should work with batches and a and b transposed', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3,2], [1,4,2,5,3,6,7,10,8,11,9,12]);
      const b = constructor([2,2,3], [7,9,11,8,10,12,13,15,17,14,16,18]);
      const expected = constructor([2, 2, 2], [58, 64, 139, 154, 364, 388, 499, 532]);

      expect(await a.gemm(b, true, true).compare(expected, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      expected.delete();
    });

    it('should work with batches and a and b transposed and c', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3,2], [1,4,2,5,3,6,7,10,8,11,9,12]);
      const b = constructor([2,2,3], [7,9,11,8,10,12,13,15,17,14,16,18]);

      const alpha = 0.5;
      const c1 = constructor([1], [1]);
      const c2 = constructor([2], [1,2]);
      const c3 = constructor([2,2], [1,2,3,4]);
      const c4 = constructor([2,2,2], [1,2,3,4,5,6,7,8]);

      const expected1 = constructor([2, 2, 2], [29 + 1,32 + 1,69.5 + 1,77 + 1,182 + 1,194 + 1,249.5 + 1,266 + 1]);
      const expected2 = constructor([2, 2, 2], [29 + 1,32 + 2,69.5 + 1,77 + 2,182 + 1,194 + 2,249.5 + 1,266 + 2]);
      const expected3 = constructor([2, 2, 2], [29 + 1,32 + 2,69.5 + 3,77 + 4,182 + 1,194 + 2,249.5 + 3,266 + 4]);
      const expected4 = constructor([2, 2, 2], [29 + 1,32 + 2,69.5 + 3,77 + 4,182 + 5,194 + 6,249.5 + 7,266 + 8]);

      expect(await a.gemm(b, true, true, alpha, c1).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.gemm(b, true, true, alpha, c2).compare(expected2, epsilon)).toBeTruthy();
      expect(await a.gemm(b, true, true, alpha, c3).compare(expected3, epsilon)).toBeTruthy();
      expect(await a.gemm(b, true, true, alpha, c4).compare(expected4, epsilon)).toBeTruthy();

      a.delete();
      b.delete();
      c1.delete();
      c2.delete();
      c3.delete();
      c4.delete();
      expected1.delete();
      expected2.delete();
      expected3.delete();
      expected4.delete();
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

  describe(`${name} transpose`, () => {
    it('should reverse the axis by default', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3,4], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]);

      const expected = constructor([4,3,2], [1,13,5,17,9,21,2,14,6,18,10,22,3,15,7,19,11,23,4,16,8,20,12,24]);

      expect(await a.transpose().compare(expected, epsilon)).toBeTruthy();

      a.delete();
      expected.delete();
    });

    it('should work for 2 swapped axis', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3,4], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]);

      const expected1 = constructor([2,4,3], [1,5,9,2,6,10,3,7,11,4,8,12,13,17,21,14,18,22,15,19,23,16,20,24]);
      const expected2 = constructor([3,2,4], [1,2,3,4,13,14,15,16,5,6,7,8,17,18,19,20,9,10,11,12,21,22,23,24]);

      expect(await a.transpose([0,2,1]).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.transpose([1,0,2]).compare(expected2, epsilon)).toBeTruthy();

      a.delete();
      expected1.delete();
      expected2.delete();
    });

    it('should work for circular transpose', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3,4], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]);

      const expected1 = constructor([3,4,2], [1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23,12,24]);

      expect(await a.transpose([1,2,0]).compare(expected1, epsilon)).toBeTruthy();

      a.delete();
      expected1.delete();
    });
  });

  describe(`${name} clip`, () => {
    it('should clip the tensor', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2, 2], [1, 2, 3, 4]);
      const expected1 = constructor([2, 2], [2, 2, 3, 3]);
      const expected2 = constructor([2, 2], [1, 2, 3, 3]);
      const expected3 = constructor([2, 2], [2, 2, 3, 4]);

      expect(await a.clip(2, 3).compare(expected1, epsilon)).toBeTruthy();
      expect(
        await a.clip(undefined, 3).compare(expected2, epsilon)
      ).toBeTruthy();
      expect(await a.clip(2).compare(expected3, epsilon)).toBeTruthy();

      a.delete();
      expected1.delete();
      expected2.delete();
      expected3.delete();
    });
  });

  describe(`${name} repeat`, () => {
    it('should reverse the axis by default', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([2,3], [1,2,3,4,5,6]);

      const expected1 = constructor([4,3], [1,2,3,4,5,6,1,2,3,4,5,6]);
      const expected2 = constructor([2,6], [1,2,3,1,2,3,4,5,6,4,5,6]);

      expect(await a.repeat([2, 1]).compare(expected1, epsilon)).toBeTruthy();
      expect(await a.repeat([1, 2]).compare(expected2, epsilon)).toBeTruthy();

      a.delete();
      expected1.delete();
      expected2.delete();
    });
  });
}
