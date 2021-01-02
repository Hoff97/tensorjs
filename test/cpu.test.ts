import testBasic from './basic';
import CPUTensor from '../js/tensor/cpu/tensor';
import testPool from './pool';

const constructor = (shape: ReadonlyArray<number>, values: number[]) => new CPUTensor(shape, values);

testBasic('CPU', constructor);
testPool('CPU', constructor);