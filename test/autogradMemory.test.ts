import {Variable} from '../lib/autograd/variable';
import Tensor from '../lib/types';
import {toGPU} from '../lib/util/convert';
import {GPUTensor} from '../lib/tensor/gpu/tensor';
import {defaultAllocator} from '../lib/tensor/gpu/gl';
import {assert} from 'console';
import {Linear, Relu, Sequential} from '../lib/model/basic';
import {SGD} from '../lib/model/optimizer';

const backends = [
  {
    name: 'GPU',
    constructor: (shape: ReadonlyArray<number>, values: number[]) =>
      new GPUTensor(new Float32Array(values), shape, 32),
    toBackend: (tensor: Tensor) => toGPU(tensor, 32),
  },
];

const run = false;

if (run) {
  for (const backend of backends) {
    describe(`Autograd on ${backend.name}`, () => {
      it('should release all memory on exp', async () => {
        const a = backend.constructor([2, 2], [-1, 0, 1, 2]);

        const v = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = v.exp().sum() as Variable;
        const res2 = v.exp().sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        v.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on log', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);

        const v = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = v.log().sum() as Variable;
        const res2 = v.log().sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        v.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on clip', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);

        const v = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = v.clip(1.5, 3.5).sum() as Variable;
        const res2 = v.clip(1.5, 3.5).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        v.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on abs', async () => {
        const a = backend.constructor([2, 2], [-1, 2, -3, 4]);

        const v = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = v.abs().sum() as Variable;
        const res2 = v.abs().sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        v.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on multiply scalar', async () => {
        const a = backend.constructor([2, 2], [-1, 2, -3, 4]);

        const v = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = v.multiplyScalar(5).sum() as Variable;
        const res2 = v.multiplyScalar(4).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        v.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on negate', async () => {
        const a = backend.constructor([2, 2], [-1, 2, -3, 4]);

        const v = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = v.negate().sum() as Variable;
        const res2 = v.negate().sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        v.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on sqrt', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);

        const v = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = v.sqrt().sum() as Variable;
        const res2 = v.sqrt().sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        v.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on add', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2, 2], [1, 2, 3, 4]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.add(vB).sum() as Variable;
        const res2 = vA.add(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on broadcasted add', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2], [1, 2]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.add(vB).sum() as Variable;
        const res2 = vA.add(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on subtract', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2, 2], [1, 2, 3, 4]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.subtract(vB).sum() as Variable;
        const res2 = vA.subtract(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on broadcasted subtract', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2], [1, 2]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.subtract(vB).sum() as Variable;
        const res2 = vA.subtract(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on divide', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2, 2], [1, 2, 3, 4]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.divide(vB).sum() as Variable;
        const res2 = vA.divide(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on broadcasted divide', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2], [1, 2]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.divide(vB).sum() as Variable;
        const res2 = vA.divide(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on multiply', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2, 2], [1, 2, 3, 4]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.multiply(vB).sum() as Variable;
        const res2 = vA.multiply(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on broadcasted multiply', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2], [1, 2]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.multiply(vB).sum() as Variable;
        const res2 = vA.multiply(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on power', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2, 2], [1, 2, 3, 4]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.power(vB).sum() as Variable;
        const res2 = vA.power(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on broadcasted power', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2], [1, 2]);

        const vA = new Variable(a);
        const vB = new Variable(b);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.power(vB).sum() as Variable;
        const res2 = vA.power(vB).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on sum', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);

        const vA = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.sum(1).sum() as Variable;
        const res2 = vA.sum(1).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on sum square', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);

        const vA = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.sumSquare(1).sum() as Variable;
        const res2 = vA.sumSquare(1).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on reshape', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);

        const vA = new Variable(a);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.reshape([4]).sum() as Variable;
        const res2 = vA.reshape([4]).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should release all memory on gemm', async () => {
        const a = backend.constructor([2, 2], [1, 2, 3, 4]);
        const b = backend.constructor([2, 3], [1, 2, 3, 4, 5, 6]);
        const c = backend.constructor([3], [1, 2, 3]);

        const vA = new Variable(a);
        const vB = new Variable(b);
        const vC = new Variable(c);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        const res1 = vA.gemm(vB, false, false, 1, vC).sum() as Variable;
        const res2 = vA.gemm(vB, false, false, 1, vC).sum() as Variable;
        res1.backward();
        res2.backward();
        res1.delete();
        res2.delete();
        vA.grad?.delete();
        vB.grad?.delete();
        vC.grad?.delete();

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });

      it('should work with models', async () => {
        const input = backend.constructor([2, 2], [1, 2, 3, 4]);
        const y = backend.constructor([2, 1], [1, 2]);

        const vInput = new Variable(input, {noGrad: true});
        const vY = new Variable(y, {noGrad: true});

        const model = new Sequential([
          new Linear(2, 4),
          new Relu(),
          new Linear(4, 1),
        ]);
        await model.toGPU(32);
        const optim = new SGD(model);

        const allocationsBefore = defaultAllocator.totalAllocations;
        const entriesBefore = defaultAllocator.getNumEntries();

        for (let i = 0; i < 5; i++) {
          const pred = model.forward(vInput);
          const loss = pred.subtract(vY).sumSquare() as Variable;
          loss.backward();
          optim.step();
          loss.delete();
          optim.zeroGrads();
        }

        const allocationsAfter = defaultAllocator.totalAllocations;
        const entriesAfter = defaultAllocator.getNumEntries();

        const allocations = allocationsAfter - allocationsBefore;
        const additionalEntries = entriesAfter - entriesBefore;

        expect(allocations).toEqual(additionalEntries);
      });
    });
  }
}
