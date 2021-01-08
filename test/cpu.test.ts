import testBasic from './basic';
import { CPUTensor } from '../lib/tensor/cpu/tensor';
import testAggregate from './aggregate';
import testConv from './conv';
import testPool from './pool';

const constructor = (shape: ReadonlyArray<number>, values: number[]) => new CPUTensor(shape, values);

testBasic('CPU', constructor);
testAggregate('CPU', constructor);
testConv('CPU', constructor);
testPool('CPU', constructor);