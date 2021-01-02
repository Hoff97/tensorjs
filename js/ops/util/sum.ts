export function sumResultShape(inputShape: readonly number[], axes: number[]) {
  const resultShape = [];
  const sumShape = [];
  const ixMap: number[] = [];
  for (let i = 0; i < inputShape.length; i++) {
    if (!axes.includes(i)) {
      resultShape.push(inputShape[i]);
      ixMap.push(i);
    } else {
      sumShape.push(inputShape[i]);
    }
  }

  if (resultShape.length === 0) {
    resultShape.push(1);
  }

  return [resultShape, ixMap];
}