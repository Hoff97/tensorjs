import {primeFactors} from '../lib/util/math';

describe('Prime factors', () => {
  it('should work', () => {
    expect(primeFactors(64)).toEqual([2, 2, 2, 2, 2, 2]);
  });
});
