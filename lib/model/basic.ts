import {Variable} from '../autograd/variable';
import {Tensor} from '../library';
import {CPUTensor} from '../tensor/cpu/tensor';
import {normal} from '../util/math';
import {Module} from './module';

/**
 * Linear layer calculates y=xW + b
 *
 * W is initialized with Xavier initialization, while the bias is
 * initialized to zeros
 */
export class Linear extends Module {
  weights: Tensor<any>;
  bias?: Tensor<any>;

  /**
   * Creates a linear layer
   * @param dimIn Feature dimension of the input
   * @param dimOut Feature dimension of the output
   * @param bias Wether a bias should be added or not. Defaults to true
   */
  constructor(dimIn: number, dimOut: number, bias?: boolean) {
    super();

    bias = bias === undefined ? true : bias;

    const weightVals = normal(dimIn * dimOut, 0, 2 / (dimIn + dimOut));
    const tensor = new CPUTensor([dimIn, dimOut], weightVals);
    this.weights = new Variable(tensor);

    if (bias) {
      const biasVals = new Array(dimOut).fill(0);
      const tensorBias = new CPUTensor([1, dimOut], biasVals);
      this.bias = new Variable(tensorBias);
    }
  }

  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    return [inputs[0].gemm(this.weights, false, false, 1, this.bias)];
  }
}

/**
 * Rectified linear unit, calculates y = max(x,0)
 */
export class Relu extends Module {
  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    return [inputs[0].clip(0)];
  }
}

/**
 * Sequence of modules. Passes the input sequentially into the specified modules
 */
export class Sequential extends Module {
  constructor(public modules: Module[]) {
    super();
  }

  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    let x = inputs;
    for (let i = 0; i < this.modules.length; i++) {
      const oldX = x;
      x = await this.modules[i].forward(x);
      if (this.mode === 'inference' && i > 0) {
        for (let j = 0; j < oldX.length; j++) {
          oldX[j].delete();
        }
      }
    }
    return x;
  }

  getSubModules(): Module[] {
    const modules = super.getSubModules();
    return modules.concat(this.modules);
  }
}

/**
 * Dictionary of modules. Use this if you want to store submodules in a dictionary
 */
export class ModuleDict extends Module {
  constructor(public modules: {[name: string]: Module} = {}) {
    super();
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    throw new Error('Module dict does not support forward');
  }

  getSubModules(): Module[] {
    const modules = [];
    for (const k in this.modules) {
      modules.push(this.modules[k]);
    }
    return modules;
  }

  get(key: string) {
    return this.modules[key];
  }

  set(key: string, module: Module) {
    this.modules[key] = module;
  }
}

/**
 * List of modules. Use this if you want to store submodules in a list
 */
export class ModuleList extends Module {
  constructor(public modules: Module[] = []) {
    super();
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    throw new Error('Module list does not support forward');
  }

  getSubModules(): Module[] {
    return this.modules;
  }

  get(index: number) {
    return this.modules[index];
  }

  set(index: number, module: Module) {
    this.modules[index] = module;
  }

  push(module: Module) {
    this.modules.push(module);
  }

  pop() {
    return this.modules.pop();
  }
}
