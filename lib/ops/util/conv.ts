export function outputDimSize(inSize: number, 
                              kernel: number,
                              headPad: number,
                              tailPad: number,
                              dilation: number,
                              stride: number) {
  const dkernel = dilation * (kernel - 1) + 1;
  return Math.floor(((inSize + headPad + tailPad - dkernel) / stride) + 1)
}

export function outputDimsSize(inSizes: number[],
                               kernels: number[],
                               headPads: number[],
                               tailPads: number[],
                               dilations: number[],
                               strides: number[]) {
  const result: number[] = [];
  for (let i = 0; i < inSizes.length; i++) {
    result.push(outputDimSize(inSizes[i], kernels[i], headPads[i], tailPads[i], dilations[i], strides[i]));
  }
  return result;
}