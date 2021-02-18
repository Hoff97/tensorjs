import {GPUTensor} from '../lib/tensor/gpu/tensor';
import {Adam} from '../lib/model/optimizer';
import {Linear} from '../lib/model/basic';
import {defaultUpdateMomentsD} from '../lib/model/optimizer/adam/updateMoments';
import {defaultUpdateValueD} from '../lib/model/optimizer/adam/updateParams';

const run = false;

if (run) {
  describe('Adam', () => {
    it('corrected moments should work', async () => {
      const moments = new GPUTensor(new Float32Array([1, 2, 3, 4]), [4], 32);
      const grad = new GPUTensor(new Float32Array([1]), [1], 32);

      const moment1 = new GPUTensor(new Float32Array([1]), [1], 32);
      const moment2 = new GPUTensor(new Float32Array([3]), [1], 32);

      const optim = new Adam(new Linear(1, 1));
      optim.t++;

      const {moment1New, moment2New} = optim.updateMoments(
        grad,
        moment1,
        moment2
      );
      const {correctMoment1, correctMoment2} = optim.getCorrectedMoments(
        moment1New,
        moment2New
      );

      const newMoments = defaultUpdateMomentsD.calc(
        {
          Grad: grad,
          Moments: moments,
          beta1: optim.beta1,
          beta2: optim.beta2,
          t: optim.t,
        },
        32
      ) as GPUTensor;
    });

    it('update step should work', async () => {
      const moments = new GPUTensor(new Float32Array([1, 2, 3, 4]), [4], 32);
      const grad = new GPUTensor(new Float32Array([1]), [1], 32);

      const value = new GPUTensor(new Float32Array([0]), [1], 32);

      const moment1 = new GPUTensor(new Float32Array([1]), [1], 32);
      const moment2 = new GPUTensor(new Float32Array([3]), [1], 32);

      const optim = new Adam(new Linear(1, 1));
      optim.t++;

      const {newValue} = optim.paramStep(value, grad, moment1, moment2);

      const newMoments = defaultUpdateMomentsD.calc(
        {
          Grad: grad,
          Moments: moments,
          beta1: optim.beta1,
          beta2: optim.beta2,
          t: optim.t,
        },
        32
      ) as GPUTensor;
      const newVal = defaultUpdateValueD.calc(
        {
          Value: value,
          Moments: newMoments,
          alpha: optim.lr,
          epsilon: optim.epsilon,
        },
        32
      ) as GPUTensor;
    });
  });
}
