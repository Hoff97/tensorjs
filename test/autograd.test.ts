import {CPUTensor} from '../lib/tensor/cpu/tensor';
import {Variable} from '../lib/autograd/variable';
import {numericalGradient} from '../lib/autograd/util/numerical';

const epsilon = 0.01;

describe('Exp backward', () => {
  it('should work', async () => {
    const a = new CPUTensor([2, 2], [-1, 0, 1, 2]);
    const ones = new CPUTensor([2, 2], [1, 1, 1, 1]);

    const v = new Variable(a);

    const res = v.exp() as Variable;
    res.backward(ones);

    const numericalGrad = numericalGradient(
      a,
      (a: CPUTensor) => a.exp() as CPUTensor
    );

    //@ts-ignore
    expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
  });
});

describe('Log backward', () => {
  it('should work', async () => {
    const a = new CPUTensor([2, 2], [-1, 0, 1, 2]);
    const ones = new CPUTensor([2, 2], [1, 1, 1, 1]);

    const v = new Variable(a);

    const res = v.log() as Variable;
    res.backward(ones);

    const numericalGrad = numericalGradient(
      a,
      (a: CPUTensor) => a.log() as CPUTensor
    );

    //@ts-ignore
    expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
  });
});

describe('Sqrt backward', () => {
  it('should work', async () => {
    const a = new CPUTensor([2, 2], [1, 2, 4, 9]);
    const ones = new CPUTensor([2, 2], [1, 1, 1, 1]);

    const v = new Variable(a);

    const res = v.sqrt() as Variable;
    res.backward(ones);

    const numericalGrad = numericalGradient(
      a,
      (a: CPUTensor) => a.sqrt() as CPUTensor
    );

    //@ts-ignore
    expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
  });
});

describe('Sqrt backward', () => {
  it('should work', async () => {
    const a = new CPUTensor([2, 2], [1, 2, 4, 9]);
    const ones = new CPUTensor([2, 2], [1, 1, 1, 1]);

    const v = new Variable(a);

    const res = v.sqrt() as Variable;
    res.backward(ones);

    const numericalGrad = numericalGradient(
      a,
      (a: CPUTensor) => a.sqrt() as CPUTensor
    );

    //@ts-ignore
    expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
  });
});

describe('Abs backward', () => {
  it('should work', async () => {
    const a = new CPUTensor([2, 2], [-2, -1, 1, 2]);
    const ones = new CPUTensor([2, 2], [1, 1, 1, 1]);

    const v = new Variable(a);

    const res = v.abs() as Variable;
    res.backward(ones);

    const numericalGrad = numericalGradient(
      a,
      (a: CPUTensor) => a.abs() as CPUTensor
    );

    //@ts-ignore
    expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
  });
});

describe('Negate backward', () => {
  it('should work', async () => {
    const a = new CPUTensor([2, 2], [-2, -1, 1, 2]);
    const ones = new CPUTensor([2, 2], [1, 1, 1, 1]);

    const v = new Variable(a);

    const res = v.negate() as Variable;
    res.backward(ones);

    const numericalGrad = numericalGradient(
      a,
      (a: CPUTensor) => a.negate() as CPUTensor
    );

    //@ts-ignore
    expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
  });
});
