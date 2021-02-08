import {CPUTensor} from '../../tensor/cpu/tensor';

const defaultEpsilon = 0.0001;

export function numericalGradient(
  input: CPUTensor,
  fun: (tensor: CPUTensor) => CPUTensor,
  epsilon?: number
) {
  if (epsilon === undefined) {
    epsilon = defaultEpsilon;
  }

  const baseVal = (fun(input).sum() as CPUTensor).get(0);

  const grad = input.constantLike(0) as CPUTensor;

  for (let i = 0; i < input.size; i++) {
    input.set(i, input.get(i) + epsilon);
    const val = (fun(input).sum() as CPUTensor).get(0);
    grad.set(i, (val - baseVal) / epsilon);
    input.set(i, input.get(i) - epsilon);
  }

  return grad;
}
