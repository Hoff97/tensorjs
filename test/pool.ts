import Tensor from '../lib/types';

const DELTA = 0.00001;

// eslint-disable-next-line no-unused-vars
type TensorConstructor = (shape: ReadonlyArray<number>, values: number[]) => Tensor

export default function testPool(name: string, constructor: TensorConstructor, wait?: Promise<void>) {
  describe(`${name} average pool`, () => {
    it('should compute the average of the whole matrix', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([1,1, 2, 2], [1, 2, 3, 4]);
      const expected = constructor([1,1,1,1], [10/4]);

      expect(await a.averagePool([2,2]).compare(expected, DELTA)).toBeTruthy();

      a.delete();
      expected.delete();
    });

    it('should compute the average of cells with stride=2', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([1,1, 4,4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
      const expected = constructor([1,1,2,2], [(1+2+5+6)/4,(3+4+7+8)/4,(9+10+13+14)/4,(11+12+15+16)/4]);

      expect(await a.averagePool([2,2], [0,0,0,0], [2,2]).compare(expected, DELTA)).toBeTruthy();

      a.delete();
      expected.delete();
    });

    it('should compute the average of cells with stride=1', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([1,1, 4,4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
      const expected = constructor([1,1,3,3], [(1+2+5+6)/4,(2+3+6+7)/4,(3+4+7+8)/4,
                                               (5+6+9+10)/4,(6+7+10+11)/4,(7+8+11+12)/4,
                                               (9+10+13+14)/4,(10+11+14+15)/4,(11+12+15+16)/4]);

      expect(await a.averagePool([2,2]).compare(expected, DELTA)).toBeTruthy();

      a.delete();
      expected.delete();
    });

    it('should respect padding', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([1,1, 2,2], [1, 2, 3, 4]);
      const expected = constructor([1,1,3,3], [1,(1+2)/2,2,2,(1+2+3+4)/4,3,3,(3+4)/2,4]);

      expect(await a.averagePool([2,2],[1,1,1,1]).compare(expected, DELTA)).toBeTruthy();

      a.delete();
      expected.delete();
    });

    it('should respect includePad = true', async () => {
      if (wait) {
        await wait;
      }

      const a = constructor([1,1, 2,2], [1, 2, 3, 4]);
      const expected = constructor([1,1,3,3], [1/4,(1+2)/4,2/4,(1+3)/4,(1+2+3+4)/4,(2+4)/4,3/4,(3+4)/4,4/4]);

      expect(await a.averagePool([2,2],[1,1,1,1],[1,1],true).compare(expected, DELTA)).toBeTruthy();

      a.delete();
      expected.delete();
    });
  });
}