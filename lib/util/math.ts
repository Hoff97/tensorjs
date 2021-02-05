export function primeFactors(num: number) {
  return primeFactorsCompute(num);
}

export function primeFactorsCompute(
  inputNum: number,
  result: number[] = [],
  repeat = true
): number[] {
  if (!Number.isInteger(inputNum)) return result;

  const num = Math.abs(inputNum);
  if (num < 2) return result;
  const sqrt = Math.sqrt(num);

  let x = 2;
  if (num % x) {
    x = 3;
    if (num % x) {
      x = 5;
      let add = 2;
      while (num % x && x < sqrt) {
        // search numbers: 5, 7, 11, 13, 17, 19, 23...
        x += add;
        // add each time: 2, 4, 2, 4, 2, 4, 2...
        add = 6 - add;
      }
    }
  }

  x = x <= sqrt ? x : num;

  if (!repeat) {
    const index = result.indexOf(x);
    if (index < 0) result.push(x);
  } else result.push(x);

  return x === num ? result : primeFactorsCompute(num / x, result, repeat);
}

/**
 * Generate two normally distributed values using the box-muller transform
 */
export function boxMuller() {
  let u1 = 0;
  let u2 = 0;
  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();

  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
  return [z0, z1];
}

/**
 * Generates n normally distributed values using the Box-Muller transform
 * @param n Number of values to generate
 * @param mean Mean of the normal distribution, defaults to 0
 * @param variance Variance of the normal distribution, defaults to 1
 */
export function normal(n: number, mean = 0, variance = 1): number[] {
  const result = [];
  for (let i = 0; i < n; i += 2) {
    const [z0, z1] = boxMuller();
    result.push(z0 * variance + mean);
    if (i < n - 1) {
      result.push(z1 * variance + mean);
    }
  }
  return result;
}
