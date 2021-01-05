import { primeFactors } from '../js/util/math';

describe('Prime factors', () => {
  it('should work', () => {
    expect(primeFactors(64)).toEqual([2,2,2,2,2,2]);
  });
});
