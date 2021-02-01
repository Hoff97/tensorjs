import {CPUTensor} from '../../tensor/cpu/tensor';

export function cast(a: CPUTensor, to: string) {
  if (a.type === to) {
    return a;
  } else if (to === 'float') {
    const arr = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      arr[i] = a.get(i);
    }
    return new CPUTensor(a.shape, arr, to);
  } else {
    const arr = new Int32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      arr[i] = a.get(i);
    }
    return new CPUTensor(a.shape, arr, to);
  }
}
