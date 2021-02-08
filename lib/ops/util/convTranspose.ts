export function outputDimSize(
  inSize: number,
  kernel: number,
  headPad: number,
  tailPad: number,
  dilation: number,
  stride: number
) {
  const kernelSize = dilation * (kernel - 1) + 1;

  return stride * (inSize - 1) + headPad + tailPad - kernelSize + 2;
}

export function outputDimsSize(
  inSizes: readonly number[],
  kernels: readonly number[],
  headPads: readonly number[],
  tailPads: readonly number[],
  dilations: readonly number[],
  strides: readonly number[]
) {
  const result: number[] = [];
  for (let i = 0; i < inSizes.length; i++) {
    result.push(
      outputDimSize(
        inSizes[i],
        kernels[i],
        headPads[i],
        tailPads[i],
        dilations[i],
        strides[i]
      )
    );
  }
  return result;
}
