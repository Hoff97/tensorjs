import {Variable} from '../autograd/variable';
import {Tensor} from '../library';
import {CPUTensor} from '../tensor/cpu/tensor';
import {normal} from '../util/math';
import {Module} from './module';

/**
 * Linear layer calculates y=xW + b
 *
 * W is initialized with Xavier initialization
 */
export class Linear extends Module {
  weights: Tensor;
  bias: Tensor;

  constructor(dimIn: number, dimOut: number) {
    super();

    const weightVals = normal(dimIn * dimOut, 0, 2 / (dimIn + dimOut));
    const tensor = new CPUTensor([dimIn, dimOut], weightVals);
    this.weights = new Variable(tensor);

    const biasVals = new Array(dimOut).fill(0);
    const tensorBias = new CPUTensor([1, dimOut], biasVals);
    this.bias = new Variable(tensorBias);
  }

  forward(input: Tensor): Tensor {
    return input.gemm(this.weights, false, false, 1, this.bias);
  }
}

export class Relu extends Module {
  forward(input: Tensor): Tensor {
    return input.clip(0);
  }
}

export class Sequential extends Module {
  constructor(public modules: Module[]) {
    super();
  }

  forward(input: Tensor): Tensor {
    let x = input;
    for (let i = 0; i < this.modules.length; i++) {
      x = this.modules[i].forward(x);
    }
    return x;
  }

  getSubModules(): Module[] {
    const modules = super.getSubModules();
    return modules.concat(this.modules);
  }
}
