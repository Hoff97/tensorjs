import {CPUTensor} from '../lib/tensor/cpu/tensor';
import {Variable} from '../lib/autograd/variable';
import {numericalGradient} from '../lib/autograd/util/numerical';
import Tensor from '../lib/types';
import {toCPU, toGPU, toWASM} from '../lib/util/convert';
import {wasmLoaded, WASMTensor} from '../lib/tensor/wasm/tensor';
import {GPUTensor} from '../lib/tensor/gpu/tensor';
import {bce} from '../lib/model/functional/bce/bce';

const epsilon = 0.01;

interface Backend {
  name: string;
  constructor: (
    shape: ReadonlyArray<number>,
    values: number[]
  ) => Tensor<'float32'>;
  toBackend: (tensor: Tensor<'float32'>) => Promise<Tensor<'float32'>>;
  wait?: Promise<void>;
}

const backends: Backend[] = [
  {
    name: 'CPU',
    constructor: (shape: ReadonlyArray<number>, values: number[]) =>
      new CPUTensor(shape, values, 'float32'),
    toBackend: (tensor: Tensor<'float32'>) => toCPU(tensor),
  },
  {
    name: 'WASM',
    constructor: (shape: ReadonlyArray<number>, values: number[]) =>
      new WASMTensor(values, new Uint32Array(shape), 'float32'),
    toBackend: (tensor: Tensor<'float32'>) => toWASM(tensor),
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

      const res = v.exp() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.exp() as CPUTensor<'float32'>
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

      const res = v.log() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.log() as CPUTensor<'float32'>
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

      const res = v.sqrt() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.sqrt() as CPUTensor<'float32'>
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

      const res = v.reshape([4]) as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.reshape([4]) as CPUTensor<'float32'>
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

      const res = v.abs() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.abs() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with sin', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-2, -1, 0.5, 1]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.sin() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.sin() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with cos', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-2, -1, 0.5, 1]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.cos() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.cos() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with tan', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-0.7, -0.3, 0.5, 0.7]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.tan() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.tan() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with asin', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-0.5, -0.1, 0.2, 0.7]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.asin() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.asin() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with acos', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-0.5, -0.1, 0.2, 0.7]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.acos() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.acos() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with atan', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-0.7, -0.3, 0.5, 0.7]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.atan() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.atan() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with sinh', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-0.5, -0.1, 0.2, 0.7]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.sinh() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.sinh() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with cosh', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-0.5, -0.1, 0.2, 0.7]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.cosh() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.cosh() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with tanh', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-0.7, -0.3, 0.5, 0.7]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.tanh() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.tanh() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    if (backend.name !== 'GPU') {
      it('should work with asinh', async () => {
        if (backend.wait !== undefined) {
          await backend.wait;
        }

        const a = backend.constructor([2, 2], [-0.5, -0.1, 0.2, 0.7]);
        const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

        const v = new Variable(a);

        const res = v.asinh() as Variable<'float32'>;
        res.backward(ones);

        const numericalGrad = await backend.toBackend(
          numericalGradient(
            (await toCPU(a)) as CPUTensor<'float32'>,
            (a: CPUTensor<'float32'>) => a.asinh() as CPUTensor<'float32'>
          )
        );

        expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
      });

      it('should work with acosh', async () => {
        if (backend.wait !== undefined) {
          await backend.wait;
        }

        const a = backend.constructor([2, 2], [2.0, 2.2, 3.1, 4.5]);
        const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

        const v = new Variable(a);

        const res = v.acosh() as Variable<'float32'>;
        res.backward(ones);

        const numericalGrad = await backend.toBackend(
          numericalGradient(
            (await toCPU(a)) as CPUTensor<'float32'>,
            (a: CPUTensor<'float32'>) => a.acosh() as CPUTensor<'float32'>
          )
        );

        expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
      });

      it('should work with atanh', async () => {
        if (backend.wait !== undefined) {
          await backend.wait;
        }

        const a = backend.constructor([2, 2], [-0.7, -0.3, 0.5, 0.7]);
        const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

        const v = new Variable(a);

        const res = v.atanh() as Variable<'float32'>;
        res.backward(ones);

        const numericalGrad = await backend.toBackend(
          numericalGradient(
            (await toCPU(a)) as CPUTensor<'float32'>,
            (a: CPUTensor<'float32'>) => a.atanh() as CPUTensor<'float32'>
          )
        );

        expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
      });
    }

    it('should work with negate', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-2, -1, 0.5, 1]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.negate() as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.negate() as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    it('should work with add multiply scalar', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 2], [-2, -1, 0.5, 1]);
      const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

      const v = new Variable(a);

      const res = v.addMultiplyScalar(2.0, 5.0) as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) =>
            a.addMultiplyScalar(2.0, 5.0) as CPUTensor<'float32'>
        )
      );

      expect(await v.grad?.compare(numericalGrad, epsilon)).toBeTrue();
    });

    if (backend.name !== 'GPU') {
      it('should work with power scalar', async () => {
        if (backend.wait !== undefined) {
          await backend.wait;
        }

        const a = backend.constructor([2, 2], [-2, -1, 0.5, 1]);
        const ones = backend.constructor([2, 2], [1, 1, 1, 1]);

        const v = new Variable(a);

        const res = v.powerScalar(2.0, 3.0) as Variable<'float32'>;
        res.backward(ones);

        const numericalGrad = await backend.toBackend(
          numericalGradient(
            (await toCPU(a)) as CPUTensor<'float32'>,
            (a: CPUTensor<'float32'>) =>
              a.powerScalar(2.0, 3.0) as CPUTensor<'float32'>
          )
        );

        expect(await v.grad?.compare(numericalGrad, 0.5)).toBeTrue();
      });
    }

    it('should work with matmul', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = new CPUTensor([2, 3], [1, 2, 3, 4, 5, 6], 'float32');
      const b = new CPUTensor(
        [3, 4],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'float32'
      );
      const ones = backend.constructor([2, 4], [1, 1, 1, 1, 1, 1, 1, 1]);

      const aBackend = await backend.toBackend(a);
      const bBackend = await backend.toBackend(b);

      const vA = new Variable(aBackend);
      const vB = new Variable(bBackend);

      const res = vA.matMul(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = numericalGradient(
        a,
        (a: CPUTensor<'float32'>) => a.matMul(b) as CPUTensor<'float32'>
      );
      const numericalGradB = numericalGradient(
        b,
        (b: CPUTensor<'float32'>) => a.matMul(b) as CPUTensor<'float32'>
      );

      expect(await vA.grad?.compare(numericalGradA, 1)).toBeTrue();
      expect(await vB.grad?.compare(numericalGradB, 1)).toBeTrue();
    });

    it('should work with concat', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = new CPUTensor([2, 3], [1, 2, 3, 4, 5, 6], 'float32');
      const b = new CPUTensor([2, 4], [1, 2, 3, 4, 5, 6, 7, 8], 'float32');
      const ones = backend.constructor(
        [2, 7],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      );

      const aBackend = await backend.toBackend(a);
      const bBackend = await backend.toBackend(b);

      const vA = new Variable(aBackend);
      const vB = new Variable(bBackend);

      const res = vA.concat(vB, 1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = numericalGradient(
        a,
        (a: CPUTensor<'float32'>) => a.concat(b, 1) as CPUTensor<'float32'>
      );
      const numericalGradB = numericalGradient(
        b,
        (b: CPUTensor<'float32'>) => a.concat(b, 1) as CPUTensor<'float32'>
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

      const res = v.clip(0, 6) as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          (await toCPU(a)) as CPUTensor<'float32'>,
          (a: CPUTensor<'float32'>) => a.clip(0, 6) as CPUTensor<'float32'>
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = v.repeat([3, 2]) as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.repeat([3, 2]) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = v.expand([3, 2, 2]) as Variable<'float32'>;
      res.backward(ones);

      const numericalGrad = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.expand([3, 2, 2]) as CPUTensor<'float32'>
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.add(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.add(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.add(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.add(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.add(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.add(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.subtract(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.subtract(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.subtract(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.subtract(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.subtract(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.subtract(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.multiply(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.multiply(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.multiply(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.multiply(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.multiply(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.multiply(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.divide(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.divide(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.divide(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.divide(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.divide(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.divide(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.power(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.power(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.power(b) as CPUTensor<'float32'>
        )
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vA.power(vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.power(bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) => aCPU.power(b) as CPUTensor<'float32'>
        )
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

      const xCPU = (await toCPU(x)) as CPUTensor<'float32'>;
      const wCPU = (await toCPU(w)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vX.conv(vW, vB) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor<'float32'>) =>
            x.conv(wCPU, bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor<'float32'>) =>
            xCPU.conv(w, bCPU) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            xCPU.conv(wCPU, b) as CPUTensor<'float32'>
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

      const xCPU = (await toCPU(x)) as CPUTensor<'float32'>;
      const wCPU = (await toCPU(w)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vX.conv(vW, vB, undefined, undefined, undefined, [
        2,
        2,
      ]) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor<'float32'>) =>
            x.conv(wCPU, bCPU, undefined, undefined, undefined, [
              2,
              2,
            ]) as CPUTensor<'float32'>
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor<'float32'>) =>
            xCPU.conv(w, bCPU, undefined, undefined, undefined, [
              2,
              2,
            ]) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            xCPU.conv(wCPU, b, undefined, undefined, undefined, [
              2,
              2,
            ]) as CPUTensor<'float32'>
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

      const xCPU = (await toCPU(x)) as CPUTensor<'float32'>;
      const wCPU = (await toCPU(w)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vX.conv(vW, vB, [2, 2]) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor<'float32'>) =>
            x.conv(wCPU, bCPU, [2, 2]) as CPUTensor<'float32'>
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor<'float32'>) =>
            xCPU.conv(w, bCPU, [2, 2]) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            xCPU.conv(wCPU, b, [2, 2]) as CPUTensor<'float32'>
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

      const xCPU = (await toCPU(x)) as CPUTensor<'float32'>;
      const wCPU = (await toCPU(w)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const res = vX.conv(vW, vB, undefined, undefined, [
        1,
        1,
        1,
        1,
      ]) as Variable<'float32'>;

      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor<'float32'>) =>
            x.conv(wCPU, bCPU, undefined, undefined, [
              1,
              1,
              1,
              1,
            ]) as CPUTensor<'float32'>
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor<'float32'>) =>
            xCPU.conv(w, bCPU, undefined, undefined, [
              1,
              1,
              1,
              1,
            ]) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            xCPU.conv(wCPU, b, undefined, undefined, [
              1,
              1,
              1,
              1,
            ]) as CPUTensor<'float32'>
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

      const xCPU = (await toCPU(x)) as CPUTensor<'float32'>;
      const wCPU = (await toCPU(w)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;

      const dil = [2, 2];
      const pads = [1, 1, 1, 1];
      const strd = [2, 2];

      const res = vX.conv(
        vW,
        vB,
        dil,
        undefined,
        pads,
        strd
      ) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor<'float32'>) =>
            x.conv(
              wCPU,
              bCPU,
              dil,
              undefined,
              pads,
              strd
            ) as CPUTensor<'float32'>
        )
      );
      const numericalGradW = await backend.toBackend(
        numericalGradient(
          wCPU,
          (w: CPUTensor<'float32'>) =>
            xCPU.conv(
              w,
              bCPU,
              dil,
              undefined,
              pads,
              strd
            ) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            xCPU.conv(
              wCPU,
              b,
              dil,
              undefined,
              pads,
              strd
            ) as CPUTensor<'float32'>
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;
      const cCPU = (await toCPU(c)) as CPUTensor<'float32'>;

      const res = vA.gemm(vB, false, false, 1, vC, 1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.gemm(bCPU, false, false, 1, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            aCPU.gemm(b, false, false, 1, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor<'float32'>) =>
            aCPU.gemm(bCPU, false, false, 1, c, 1) as CPUTensor<'float32'>
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;
      const cCPU = (await toCPU(c)) as CPUTensor<'float32'>;

      const res = vA.gemm(vB, true, false, 1, vC, 1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.gemm(bCPU, true, false, 1, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            aCPU.gemm(b, true, false, 1, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor<'float32'>) =>
            aCPU.gemm(bCPU, true, false, 1, c, 1) as CPUTensor<'float32'>
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;
      const cCPU = (await toCPU(c)) as CPUTensor<'float32'>;

      const res = vA.gemm(vB, false, true, 1, vC, 1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.gemm(bCPU, false, true, 1, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            aCPU.gemm(b, false, true, 1, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor<'float32'>) =>
            aCPU.gemm(bCPU, false, true, 1, c, 1) as CPUTensor<'float32'>
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;
      const cCPU = (await toCPU(c)) as CPUTensor<'float32'>;

      const res = vA.gemm(vB, true, true, 1, vC, 1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.gemm(bCPU, true, true, 1, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            aCPU.gemm(b, true, true, 1, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor<'float32'>) =>
            aCPU.gemm(bCPU, true, true, 1, c, 1) as CPUTensor<'float32'>
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;
      const cCPU = (await toCPU(c)) as CPUTensor<'float32'>;

      const res = vA.gemm(vB, true, true, 0.5, vC, 1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.gemm(bCPU, true, true, 0.5, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            aCPU.gemm(b, true, true, 0.5, cCPU, 1) as CPUTensor<'float32'>
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor<'float32'>) =>
            aCPU.gemm(bCPU, true, true, 0.5, c, 1) as CPUTensor<'float32'>
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

      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;
      const bCPU = (await toCPU(b)) as CPUTensor<'float32'>;
      const cCPU = (await toCPU(c)) as CPUTensor<'float32'>;

      const res = vA.gemm(vB, true, true, 1, vC, 0.5) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.gemm(bCPU, true, true, 1, cCPU, 0.5) as CPUTensor<'float32'>
        )
      );
      const numericalGradB = await backend.toBackend(
        numericalGradient(
          bCPU,
          (b: CPUTensor<'float32'>) =>
            aCPU.gemm(b, true, true, 1, cCPU, 0.5) as CPUTensor<'float32'>
        )
      );
      const numericalGradC = await backend.toBackend(
        numericalGradient(
          cCPU,
          (c: CPUTensor<'float32'>) =>
            aCPU.gemm(bCPU, true, true, 1, c, 0.5) as CPUTensor<'float32'>
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const permutation = [2, 0, 1];

      const res = vA.transpose(permutation) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.transpose(permutation) as CPUTensor<'float32'>
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sum(1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.sum(1) as CPUTensor<'float32'>
        )
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sum(0) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.sum(0) as CPUTensor<'float32'>
        )
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sum() as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.sum() as CPUTensor<'float32'>
        )
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sum(undefined, true) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.sum(undefined, true) as CPUTensor<'float32'>
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sumSquare(1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.sumSquare(1) as CPUTensor<'float32'>
        )
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sumSquare(0) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.sumSquare(0) as CPUTensor<'float32'>
        )
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sumSquare() as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.sumSquare() as CPUTensor<'float32'>
        )
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
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sumSquare(undefined, true) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.sumSquare(undefined, true) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.1)).toBeTrue();
    });

    it('should work with mean axis 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([2], new Array(2).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceMean(1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.reduceMean(1) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with mean axis 0', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([3], new Array(3).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceMean(0) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.reduceMean(0) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with mean across all axes', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceMean() as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.reduceMean() as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with mean square axis 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([2], new Array(2).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceMeanSquare(1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.reduceMeanSquare(1) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with mean square axis 0', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([3], new Array(3).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceMeanSquare(0) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.reduceMeanSquare(0) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with mean square across all axes', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceMeanSquare() as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.reduceMeanSquare() as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with log sum axis 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([2], new Array(2).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceLogSum(1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.reduceLogSum(1) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with log sum axis 0', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([3], new Array(3).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceLogSum(0) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.reduceLogSum(0) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with log sum across all axes', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceLogSum() as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.reduceLogSum() as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with log sum exp axis 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([2], new Array(2).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceLogSumExp(1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.reduceLogSumExp(1) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with log sum exp axis 0', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([3], new Array(3).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceLogSumExp(0) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.reduceLogSumExp(0) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with log sum exp across all axes', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.reduceLogSumExp() as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.reduceLogSumExp() as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with slice', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3, 4], new Array(24).fill(5));
      const ones = backend.constructor([2, 1, 3], new Array(6).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.slice([2, 0], [3, 3], [1, 2]) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) =>
            a.slice([2, 0], [3, 3], [1, 2]) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.01)).toBeTrue();
    });

    it('should work with product axis 1', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([2], new Array(2).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.product(1) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.product(1) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with product axis 0', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([3], new Array(3).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.product(0) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.product(0) as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work with product across all axes', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
      const ones = backend.constructor([1], new Array(1).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.product() as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.product() as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.8)).toBeTrue();
    });

    it('should work with sigmoid', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const a = backend.constructor([2, 3], [-1, 2, -3, 4, -5, 6]);
      const ones = backend.constructor([2, 3], new Array(6).fill(1));
      const vA = new Variable(a);
      const aCPU = (await toCPU(a)) as CPUTensor<'float32'>;

      const res = vA.sigmoid() as Variable<'float32'>;
      res.backward(ones);

      const numericalGradA = await backend.toBackend(
        numericalGradient(
          aCPU,
          (a: CPUTensor<'float32'>) => a.sigmoid() as CPUTensor<'float32'>
        )
      );

      expect(await vA.grad?.compare(numericalGradA, 0.05)).toBeTrue();
    });

    it('should work for bce', async () => {
      if (backend.wait !== undefined) {
        await backend.wait;
      }

      const x = backend.constructor([2, 2], [0.4, 0.5, 0.6, 0.7]);
      const y = backend.constructor([2, 2], [1, 0, 1, 0]);
      const ones = backend.constructor([2, 2], new Array(4).fill(1));

      const vX = new Variable(x);
      const vY = new Variable(y);

      const xCPU = (await toCPU(x)) as CPUTensor<'float32'>;
      const yCPU = (await toCPU(y)) as CPUTensor<'float32'>;

      const res = bce(vX, vY) as Variable<'float32'>;
      res.backward(ones);

      const numericalGradX = await backend.toBackend(
        numericalGradient(
          xCPU,
          (x: CPUTensor<'float32'>) => bce(x, yCPU) as CPUTensor<'float32'>
        )
      );

      expect(await vX.grad?.compare(numericalGradX, 0.01)).toBeTrue();
    });
  });
}
