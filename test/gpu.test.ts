import GPUTensor from "../js/tensor/gpu/tensor";
import testBasic from './basic';
import testConv from "./conv";
import testAggregate from './aggregate';

describe(`GPU create tensor`, () => {
  it('should get the same values back', async () => {
    const arr = new Float32Array([1.,2.,3.,4.]);
    const tensor = new GPUTensor(arr, [2,2]);

    expect(await tensor.getValues()).toEqual(arr);
  });
});

const constructor = (shape: ReadonlyArray<number>, values: number[]) => {
  const vals = Float32Array.from(values);
  return new GPUTensor(vals, shape);
};

/*testBasic('GPU', constructor);
testPool('GPU', constructor);
testConv('GPU', constructor);*/