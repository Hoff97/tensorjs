import {CPUTensor} from '../lib/tensor/cpu/tensor';
import {Variable} from '../lib/autograd/variable';
import {numericalGradient} from '../lib/autograd/util/numerical';
import Tensor from '../lib/types';
import {toCPU, toGPU, toWASM} from '../lib/util/convert';
import {wasmLoaded, WASMTensor} from '../lib/tensor/wasm/tensor';
import {GPUTensor} from '../lib/tensor/gpu/tensor';

const epsilon = 0.01;

interface Backend {
  name: string;
  constructor: (shape: ReadonlyArray<number>, values: number[]) => Tensor;
  toBackend: (tensor: Tensor) => Promise<Tensor>;
  wait?: Promise<void>;
}

const backends: Backend[] = [
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
  } /*,
  {
    name: 'GPU',
    constructor: (shape: ReadonlyArray<number>, values: number[]) =>
      new GPUTensor(new Float32Array(values), shape, 32),
    toBackend: (tensor: Tensor) => toGPU(tensor, 32),
  }*/,
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
          (await toCPU(a)) as CPUTensor,
          (a: CPUTensor) => a.exp() as CPUTensor
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
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
          (await toCPU(a)) as CPUTensor,
          (a: CPUTensor) => a.log() as CPUTensor
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
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
          (await toCPU(a)) as CPUTensor,
          (a: CPUTensor) => a.sqrt() as CPUTensor
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
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
          (await toCPU(a)) as CPUTensor,
          (a: CPUTensor) => a.reshape([4]) as CPUTensor
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
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
          (await toCPU(a)) as CPUTensor,
          (a: CPUTensor) => a.abs() as CPUTensor
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
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
          (await toCPU(a)) as CPUTensor,
          (a: CPUTensor) => a.negate() as CPUTensor
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
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

      expect(await vA.grad?.compare(numericalGradA, 1)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 1)).toBeTrue();
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

      expect(await vA.grad?.compare(numericalGradA, epsilon)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, epsilon)).toBeTrue();
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
          (await toCPU(a)) as CPUTensor,
          (a: CPUTensor) => a.clip(0, 6) as CPUTensor
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with repeat', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const ones = backend.constructor([6, 4], new Array(24).fill(1));

      const v = new Variable(a);

      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = v.repeat([3, 2]) as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.repeat([3, 2]) as CPUTensor)
      );

      expect(await v.grad?.compare(numericalGrad, 0.1)).toBeTrue();
    });

    it('should work with expand', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const ones = backend.constructor([3, 2, 2], new Array(12).fill(1));

      const v = new Variable(a);

      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = v.expand([3, 2, 2]) as Variable;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.expand([3, 2, 2]) as CPUTensor
        )
      );

      expect(await v.grad?.compare(numericalGrad, 0.1)).toBeTrue();
    });

    it('should work with add', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2, 2], [5, 6, 7, 8]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.add(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.add(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.add(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, epsilon)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, epsilon)).toBeTrue();
    });

    it('should work with broadcasted add', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2], [5, 6]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.add(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.add(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.add(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.05)).toBeTrue();
    });

    it('should work with subtract', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2, 2], [5, 6, 7, 8]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.subtract(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.subtract(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.subtract(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, epsilon)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, epsilon)).toBeTrue();
    });

    it('should work with broadcasted subtract', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2], [5, 6]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.subtract(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.subtract(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.subtract(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.05)).toBeTrue();
    });

    it('should work with multiply', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2, 2], [5, 6, 7, 8]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.multiply(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.multiply(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.multiply(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.1)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.1)).toBeTrue();
    });

    it('should work with broadcasted multiply', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2], [5, 6]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.multiply(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.multiply(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.multiply(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.05)).toBeTrue();
    });

    it('should work with divide', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2, 2], [5, 6, 7, 8]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.divide(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.divide(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.divide(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.05)).toBeTrue();
    });

    it('should work with broadcasted divide', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2], [5, 6]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.divide(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.divide(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.divide(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.05)).toBeTrue();
    });

    it('should work with power', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2, 2], [1.1, 2.2, 3.3, 2.5]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.power(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.power(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.power(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
    });

    it('should work with broadcasted power', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([2], [1.5, 2.5]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vA.power(vB) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.power(bCPU) as CPUTensor)
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(bCPU, (b: CPUTensor) => aCPU.power(b) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.05)).toBeTrue();
    });

    it('should work with convolution', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const x = backend.constructor([1, 1, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const w = backend.constructor([1, 1, 2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([1], [5]);
      const ones = backend.constructor([1, 1, 2, 2], new Array(4).fill(1));

      const vX = new Variable(x);
      const vW = new Variable(w);
      const vB = new Variable(b);

      const xCPU = (await toCPU(x)) as CPUTensor;
      const wCPU = (await toCPU(w)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vX.conv(vW, vB) as Variable;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor) => x.conv(wCPU, bCPU) as CPUTensor
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor) => xCPU.conv(w, bCPU) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) => xCPU.conv(wCPU, b) as CPUTensor
        )
      );

      expect(await vX.grad?.compare(numericalGradX, 0.5)).toBeTrue();
      expect(await vW.grad?.compare(numericalGradW, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
    });

    it('should work with strided convolution', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const x = backend.constructor(
        [1, 1, 4, 4],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
      );
      const w = backend.constructor([1, 1, 2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([1], [5]);
      const ones = backend.constructor([1, 1, 2, 2], new Array(4).fill(1));

      const vX = new Variable(x);
      const vW = new Variable(w);
      const vB = new Variable(b);

      const xCPU = (await toCPU(x)) as CPUTensor;
      const wCPU = (await toCPU(w)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vX.conv(vW, vB, undefined, undefined, undefined, [
        2,
        2,
      ]) as Variable;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor) =>
            x.conv(wCPU, bCPU, undefined, undefined, undefined, [
              2,
              2,
            ]) as CPUTensor
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor) =>
            xCPU.conv(w, bCPU, undefined, undefined, undefined, [
              2,
              2,
            ]) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) =>
            xCPU.conv(wCPU, b, undefined, undefined, undefined, [
              2,
              2,
            ]) as CPUTensor
        )
      );

      expect(await vX.grad?.compare(numericalGradX, 0.5)).toBeTrue();
      expect(await vW.grad?.compare(numericalGradW, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
    });

    it('should work with dilated convolution', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const x = backend.constructor(
        [1, 1, 4, 4],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
      );
      const w = backend.constructor([1, 1, 2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([1], [5]);
      const ones = backend.constructor([1, 1, 2, 2], new Array(4).fill(1));

      const vX = new Variable(x);
      const vW = new Variable(w);
      const vB = new Variable(b);

      const xCPU = (await toCPU(x)) as CPUTensor;
      const wCPU = (await toCPU(w)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vX.conv(vW, vB, [2, 2]) as Variable;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor) => x.conv(wCPU, bCPU, [2, 2]) as CPUTensor
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor) => xCPU.conv(w, bCPU, [2, 2]) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) => xCPU.conv(wCPU, b, [2, 2]) as CPUTensor
        )
      );

      expect(await vX.grad?.compare(numericalGradX, 0.5)).toBeTrue();
      expect(await vW.grad?.compare(numericalGradW, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
    });

    it('should work with padded convolution', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const x = backend.constructor([1, 1, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const w = backend.constructor([1, 1, 2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([1], [5]);
      const ones = backend.constructor([1, 1, 4, 4], new Array(16).fill(1));

      const vX = new Variable(x);
      const vW = new Variable(w);
      const vB = new Variable(b);

      const xCPU = (await toCPU(x)) as CPUTensor;
      const wCPU = (await toCPU(w)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const res = vX.conv(vW, vB, undefined, undefined, [
        1,
        1,
        1,
        1,
      ]) as Variable;

      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor) =>
            x.conv(wCPU, bCPU, undefined, undefined, [1, 1, 1, 1]) as CPUTensor
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor) =>
            xCPU.conv(w, bCPU, undefined, undefined, [1, 1, 1, 1]) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) =>
            xCPU.conv(wCPU, b, undefined, undefined, [1, 1, 1, 1]) as CPUTensor
        )
      );

      expect(await vX.grad?.compare(numericalGradX, 0.5)).toBeTrue();
      expect(await vW.grad?.compare(numericalGradW, 0.8)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
    });

    it('should work with padded strided dilated convolution', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const x = backend.constructor([1, 1, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const w = backend.constructor([1, 1, 2, 2], [1, 2, 3, 4]);
      const b = backend.constructor([1], [5]);
      const ones = backend.constructor([1, 1, 2, 2], new Array(4).fill(1));

      const vX = new Variable(x);
      const vW = new Variable(w);
      const vB = new Variable(b);

      const xCPU = (await toCPU(x)) as CPUTensor;
      const wCPU = (await toCPU(w)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;

      const dil = [2, 2];
      const pads = [1, 1, 1, 1];
      const strd = [2, 2];

      const res = vX.conv(vW, vB, dil, undefined, pads, strd) as Variable;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor) =>
            x.conv(wCPU, bCPU, dil, undefined, pads, strd) as CPUTensor
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor) =>
            xCPU.conv(w, bCPU, dil, undefined, pads, strd) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) =>
            xCPU.conv(wCPU, b, dil, undefined, pads, strd) as CPUTensor
        )
      );

      expect(await vX.grad?.compare(numericalGradX, 0.5)).toBeTrue();
      expect(await vW.grad?.compare(numericalGradW, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
    });

    it('should work gemm', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const b = backend.constructor(
        [2, 2, 3],
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
      );
      const c = backend.constructor([3], [5, 6, 7]);
      const ones = backend.constructor([2, 2, 3], new Array(12).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);
      const vC = new Variable(c);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;
      const cCPU = (await toCPU(c)) as CPUTensor;

      const res = vA.gemm(vB, false, false, 1, vC, 1) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.gemm(bCPU, false, false, 1, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) => aCPU.gemm(b, false, false, 1, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor) => aCPU.gemm(bCPU, false, false, 1, c, 1) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
      expect(await vC.grad?.compare(numericalGradC, 0.1)).toBeTrue();
    });

    it('should work gemm a transposed', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const b = backend.constructor(
        [2, 2, 3],
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
      );
      const c = backend.constructor([3], [5, 6, 7]);
      const ones = backend.constructor([2, 2, 3], new Array(12).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);
      const vC = new Variable(c);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;
      const cCPU = (await toCPU(c)) as CPUTensor;

      const res = vA.gemm(vB, true, false, 1, vC, 1) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.gemm(bCPU, true, false, 1, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) => aCPU.gemm(b, true, false, 1, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor) => aCPU.gemm(bCPU, true, false, 1, c, 1) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
      expect(await vC.grad?.compare(numericalGradC, 0.1)).toBeTrue();
    });

    it('should work gemm b transposed', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const b = backend.constructor(
        [2, 3, 2],
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
      );
      const c = backend.constructor([3], [5, 6, 7]);
      const ones = backend.constructor([2, 2, 3], new Array(12).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);
      const vC = new Variable(c);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;
      const cCPU = (await toCPU(c)) as CPUTensor;

      const res = vA.gemm(vB, false, true, 1, vC, 1) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.gemm(bCPU, false, true, 1, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) => aCPU.gemm(b, false, true, 1, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor) => aCPU.gemm(bCPU, false, true, 1, c, 1) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
      expect(await vC.grad?.compare(numericalGradC, 0.1)).toBeTrue();
    });

    it('should work gemm a and b transposed', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const b = backend.constructor(
        [2, 3, 2],
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
      );
      const c = backend.constructor([3], [5, 6, 7]);
      const ones = backend.constructor([2, 2, 3], new Array(12).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);
      const vC = new Variable(c);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;
      const cCPU = (await toCPU(c)) as CPUTensor;

      const res = vA.gemm(vB, true, true, 1, vC, 1) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.gemm(bCPU, true, true, 1, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) => aCPU.gemm(b, true, true, 1, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor) => aCPU.gemm(bCPU, true, true, 1, c, 1) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
      expect(await vC.grad?.compare(numericalGradC, 0.1)).toBeTrue();
    });

    it('should work gemm alpha=0.5', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const b = backend.constructor(
        [2, 3, 2],
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
      );
      const c = backend.constructor([3], [5, 6, 7]);
      const ones = backend.constructor([2, 2, 3], new Array(12).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);
      const vC = new Variable(c);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;
      const cCPU = (await toCPU(c)) as CPUTensor;

      const res = vA.gemm(vB, true, true, 0.5, vC, 1) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.gemm(bCPU, true, true, 0.5, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) => aCPU.gemm(b, true, true, 0.5, cCPU, 1) as CPUTensor
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor) => aCPU.gemm(bCPU, true, true, 0.5, c, 1) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
      expect(await vC.grad?.compare(numericalGradC, 0.1)).toBeTrue();
    });

    it('should work gemm beta=0.5', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      const b = backend.constructor(
        [2, 3, 2],
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
      );
      const c = backend.constructor([3], [5, 6, 7]);
      const ones = backend.constructor([2, 2, 3], new Array(12).fill(1));

      const vA = new Variable(a);
      const vB = new Variable(b);
      const vC = new Variable(c);

      const aCPU = (await toCPU(a)) as CPUTensor;
      const bCPU = (await toCPU(b)) as CPUTensor;
      const cCPU = (await toCPU(c)) as CPUTensor;

      const res = vA.gemm(vB, true, true, 1, vC, 0.5) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.gemm(bCPU, true, true, 1, cCPU, 0.5) as CPUTensor
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor) => aCPU.gemm(b, true, true, 1, cCPU, 0.5) as CPUTensor
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor) => aCPU.gemm(bCPU, true, true, 1, c, 0.5) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.5)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 0.5)).toBeTrue();
      expect(await vC.grad?.compare(numericalGradC, 0.3)).toBeTrue();
    });

    it('should work with transpose', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3, 4], new Array(24).fill(5));
      const ones = backend.constructor([4, 2, 3], new Array(24).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const permutation = [2, 0, 1];

      const res = vA.transpose(permutation) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.transpose(permutation) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with sum axis 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([2], new Array(2).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = vA.sum(1) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.sum(1) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with sum axis 0', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([3], new Array(3).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = vA.sum(0) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.sum(0) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with sum across all axes', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = vA.sum() as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.sum() as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with sum keepDims = True', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([1, 1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = vA.sum(undefined, true) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.sum(undefined, true) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with sum square axis 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, -2, 3, -4, -5, 6]);
      const ones = backend.constructor([2], new Array(2).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = vA.sumSquare(1) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.sumSquare(1) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.1)).toBeTrue();
    });

    it('should work with sum square axis 0', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, -2, 3, -4, -5, 6]);
      const ones = backend.constructor([3], new Array(3).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = vA.sumSquare(0) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.sumSquare(0) as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.1)).toBeTrue();
    });

    it('should work with sum square across all axes', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, -2, 3, -4, -5, 6]);
      const ones = backend.constructor([1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = vA.sumSquare() as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(aCPU, (a: CPUTensor) => a.sumSquare() as CPUTensor)
      );

      expect(await vA.grad?.compare(numericalGradA, 0.1)).toBeTrue();
    });

    it('should work with sum square keepDims = True', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, -2, 3, -4, -5, 6]);
      const ones = backend.constructor([1, 1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor;

      const res = vA.sumSquare(undefined, true) as Variable;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor) => a.sumSquare(undefined, true) as CPUTensor
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.1)).toBeTrue();
    });
  });
}
