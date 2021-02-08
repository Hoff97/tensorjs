import {Variable} from '../autograd/variable';
import Tensor, {Precision} from '../types';
import {Backend, toCPU, toWASM, toGPU} from '../util/convert';

export type Mode = 'train' | 'inference';

/**
 * A module is a self contained unit that transforms
 * a list of inputs when forward is called.
 *
 * It can be in two modes, training and inference.
 * In training mode, gradients will be tracked, while
 * in inference mode, only the forward pass will be calculated
 */
export abstract class Module {
  public backend: Backend = 'CPU';

  public mode: Mode = 'train';

  abstract forward(inputs: Tensor[]): Promise<Tensor[]>;

  getSubModules(): Module[] {
    const modules: Module[] = [];
    for (const k of Object.keys(this)) {
      //@ts-ignore
      if (this[k] instanceof Module) {
        //@ts-ignore
        modules.push(this[k]);
      }
    }
    return modules;
  }

  getParameters(): Variable[] {
    let parameters: Variable[] = [];

    for (const k of Object.keys(this)) {
      //@ts-ignore
      if (this[k] instanceof Variable) {
        //@ts-ignore
        parameters.push(this[k]);
      }
    }

    const modules = this.getSubModules();
    for (const module of modules) {
      const params = module.getParameters();
      parameters = parameters.concat(params);
    }

    return parameters;
  }

  toBackend(backend: Backend) {
    if (backend === 'CPU') {
      return this.toCPU();
    } else if (backend === 'WASM') {
      return this.toWASM();
    } else {
      return this.toGPU(32);
    }
  }

  async toCPU() {
    const submodules = this.getSubModules();
    for (const submodule of submodules) {
      await submodule.toCPU();
    }

    for (const k of Object.keys(this)) {
      //@ts-ignore
      if (this[k] instanceof Tensor) {
        //@ts-ignore
        this[k] = await toCPU(this[k]);
      }
    }

    this.backend = 'CPU';
  }

  async toWASM() {
    const submodules = this.getSubModules();
    for (const submodule of submodules) {
      await submodule.toWASM();
    }

    for (const k of Object.keys(this)) {
      //@ts-ignore
      if (this[k] instanceof Tensor) {
        //@ts-ignore
        this[k] = await toWASM(this[k]);
      }
    }

    this.backend = 'WASM';
  }

  async toGPU(precision: Precision) {
    const submodules = this.getSubModules();
    for (const submodule of submodules) {
      await submodule.toGPU(precision);
    }

    for (const k of Object.keys(this)) {
      //@ts-ignore
      if (this[k] instanceof Tensor) {
        //@ts-ignore
        this[k] = await toGPU(this[k], precision);
      }
    }

    this.backend = 'GPU';
  }
}
