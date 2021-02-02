import {CPUTensor} from '../lib/tensor/cpu/tensor';
import {Variable} from '../lib/autograd/variable';
import {numericalGradient} from '../lib/autograd/util/numerical';
import Tensor from '../lib/types';
import {toCPU, toGPU, toWASM} from '../lib/util/convert';
import {wasmLoaded, WASMTensor} from '../lib/tensor/wasm/tensor';
import {GPUTensor} from '../lib/tensor/gpu/tensor';

const epsilon = 0.01;

const backends = [
  {
    name: 'CPU',
    constructor: (shape: ReadonlyArray<number>, values: number[]) =>
      new CPUTensor(shape, values),
    toBackend: (tensor: Tensor) => toCPU(tensor),
  },
  {
    name: 'WASM',
    constructor: (shape: ReadonlyArray<number>, values: number[]) =>
      new WASMTensor(new Float32Array(values), new Uint32Array(shape)),
    toBackend: (tensor: Tensor) => toWASM(tensor),
    wait: wasmLoaded,
  },
  {
    name: 'GPU',
    constructor: (shape: ReadonlyArray<number>, values: number[]) =>
      new GPUTensor(new Float32Array(values), shape, 32),
    toBackend: (tensor: Tensor) => toGPU(tensor, 32),
  },
];

for (const backend of backends) {
  describe(`Autograd on ${backend.name}`, () => {
    it('should work with exp', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-1, 0, 1, 2]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.exp() as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          await toCPU(a),
          (a: CPUTensor) => a.exp() as CPUTensor
        )
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with log', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.log() as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          await toCPU(a),
          (a: CPUTensor) => a.log() as CPUTensor
        )
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with sqrt', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 4, 16]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.sqrt() as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          await toCPU(a),
          (a: CPUTensor) => a.sqrt() as CPUTensor
        )
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with reshape', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 4, 16]);
      const ones = backend.constructor([4], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.reshape([4]) as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          await toCPU(a),
          (a: CPUTensor) => a.reshape([4]) as CPUTensor
        )
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with abs', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-2, -1, 0.5, 1]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.abs() as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          await toCPU(a),
          (a: CPUTensor) => a.abs() as CPUTensor
        )
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with negate', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-2, -1, 0.5, 1]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.negate() as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          await toCPU(a),
          (a: CPUTensor) => a.negate() as CPUTensor
        )
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with matmul', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = new CPUTensor([2, 3], [1, 2, 3, 4, 5, 6]);
      const b = new CPUTensor([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
      const ones = backend.constructor([2, 4], [1, 1, 1, 1, 1, 1, 1, 1]);

      const aBackend = await backend.toBackend(a);
      const bBackend = await backend.toBackend(b);

      const vA = new Variable(aBackend);
      const vB = new Variable(bBackend);

      const res = vA.matMul(vB) as Variable;
      res.backward(ones);

      const numericalGradA = numericalGradient(
        a,
        (a: CPUTensor) => a.matMul(b) as CPUTensor
      );
      const numericalGradB = numericalGradient(
        b,
        (b: CPUTensor) => a.matMul(b) as CPUTensor
      );

      //@ts-ignore
      expect(await vA.grad.compare(numericalGradA, 1)).toBeTrue();
      //@ts-ignore
      expect(await vB.grad.compare(numericalGradB, 1)).toBeTrue();
    });

    it('should work with concat', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = new CPUTensor([2, 3], [1, 2, 3, 4, 5, 6]);
      const b = new CPUTensor([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
      const ones = backend.constructor(
        [2, 7],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      );

      const aBackend = await backend.toBackend(a);
      const bBackend = await backend.toBackend(b);

      const vA = new Variable(aBackend);
      const vB = new Variable(bBackend);

      const res = vA.concat(vB, 1) as Variable;
      res.backward(ones);

      const numericalGradA = numericalGradient(
        a,
        (a: CPUTensor) => a.concat(b, 1) as CPUTensor
      );
      const numericalGradB = numericalGradient(
        b,
        (b: CPUTensor) => a.concat(b, 1) as CPUTensor
      );

      //@ts-ignore
      expect(await vA.grad.compare(numericalGradA, epsilon)).toBeTrue();
      //@ts-ignore
      expect(await vB.grad.compare(numericalGradB, epsilon)).toBeTrue();
    });

    it('should work with clip', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [-2, -1, 0.5, 1, 5.5, 7]);
      const ones = backend.constructor([2, 3], [1, 1, 1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.clip(0, 6) as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          await toCPU(a),
          (a: CPUTensor) => a.clip(0, 6) as CPUTensor
        )
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with repeat', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const ones = backend.constructor([6, 4], new Array(24).fill(1));

      const v = new Variable(a);

      const aCPU = await toCPU(a);

      const res = v.repeat([3, 2]) as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.repeat([3, 2]) as CPUTensor)
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, 0.1)).toBeTrue();
    });

    it('should work with expand', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const ones = backend.constructor([3, 2, 2], new Array(12).fill(1));

      const v = new Variable(a);

      const aCPU = await toCPU(a);

      const res = v.expand([3, 2, 2]) as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.expand([3, 2, 2]) as CPUTensor
        )
      );

      //@ts-ignore
      expect(await v.grad.compare(numericalGrad, 0.1)).toBeTrue();
    });
  });
}
