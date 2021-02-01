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
