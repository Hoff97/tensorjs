export function poolResultShape(inputShape: readonly number[], axes: number[], keepDims: boolean) {
  const resultShape = [];
  const sumShape = [];
  const ixMap: number[] = [];
  for (let i = 0; i < inputShape.length; i++) {
    if (!axes.includes(i)) {
      resultShape.push(inputShape[i]);
      ixMap.push(i);
    } else {
      if (keepDims) {
        resultShape.push(1);
        ixMap.push(i);
      }
      sumShape.push(inputShape[i]);
    }
  }

  if (resultShape.length === 0) {
    resultShape.push(1);
  }

  return [resultShape, ixMap];
}