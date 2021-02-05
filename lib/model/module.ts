import {Variable} from '../autograd/variable';
import Tensor from '../types';
import {Backend, toCPU} from '../util/convert';

export abstract class Module {
  public backend: Backend = 'CPU';

  abstract forward(input: Tensor): Tensor;

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

  async toGPU() {
    const submodules = this.getSubModules();
    for (const submodule of submodules) {
      await submodule.toGPU();
    }

    for (const k of Object.keys(this)) {
      //@ts-ignore
      if (this[k] instanceof Tensor) {
        //@ts-ignore
        this[k] = await toGPU(this[k]);
      }
    }

    this.backend = 'GPU';
  }
}
