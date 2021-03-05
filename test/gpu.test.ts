/* eslint-disable @typescript-eslint/no-unused-vars */
import {GPUTensor} from '../lib/tensor/gpu/tensor';
import testBasic from './basic';
import testConv from './conv';
import testAggregate from './aggregate';
import testPool from './pool';

describe('GPU create tensor', () => {
  it('should get the same values back', async () => {
    const arr = [1, 2, 3, 4];
    const tensor = new GPUTensor(arr, [2, 2], 'float32');

    expect(await tensor.getValues()).toEqual(arr);
  });

  it('should get the same values back when using half precision', async () => {
    const arr = [1, 2, 3, 4];
    const tensor = new GPUTensor(arr, [2, 2], 'float16');

    expect(await tensor.copy().getValues()).toEqual(arr);
  });
});

const constructor = (shape: ReadonlyArray<number>, values: number[]) => {
  return new GPUTensor(values, shape, 'float32');
};

/*testBasic('GPU', constructor);
testAggregate('GPU', constructor);
testPool('GPU', constructor);
testConv('GPU', constructor);*/
