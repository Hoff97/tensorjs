import testBasic from './basic';
import CPUTensor from '../js/tensor/cpu/tensor';

testBasic('CPU', (shape: ReadonlyArray<number>, values: number[]) => new CPUTensor(shape, values));
