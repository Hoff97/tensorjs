export function getSize(shape: ReadonlyArray<number>) {
  if (shape.length === 0) {
    return 0;
  }

  let size = 1;
  for (let i = 0; i < shape.length; i += 1) {
    size *= shape[i];
  }
  return size;
}

export function computeStrides(shape: ReadonlyArray<number>) {
  const rank = shape.length;

  if (rank === 0) {
    return [];
  }
  if (rank === 1) {
    return [1];
  }

  const strides = new Array(rank);
  strides[rank - 1] = 1;
  for (let i = rank - 2; i >= 0; i -= 1) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  return strides;
}

export function indexToPos(index: ReadonlyArray<number>, strides: ReadonlyArray<number>) {
  let ix = 0;
  for (let i = 0; i < index.length; i += 1) {
    ix += index[i] * strides[i];
  }
  return ix;
}

export function posToIndex(pos: number, strides: ReadonlyArray<number>): number[] {
  let res = pos;
  const rank = strides.length;
  const index = new Array(rank);

  for (let i = 0; i < index.length; i += 1) {
    index[i] = Math.floor(res / strides[i]);
    res %= strides[i];
  }

  return index;
}

export function compareShapes(a: ReadonlyArray<number>, b: ReadonlyArray<number>) {
  if (a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }

  return true;
}

export function incrementIndex(index: number[], shape: readonly number[]) {
  for (let i = index.length - 1; i >= 0; i--) {
    index[i] += 1;
    if (index[i] >= shape[i]) {
      index[i] = 0;
    } else {
      break;
    }
  }
}

export function decrementIndex(index: number[], shape: readonly number[]) {
  for (let i = index.length - 1; i >= 0; i--) {
    index[i] -= 1;
    if (index[i] < 0) {
      index[i] = shape[i] - 1;
    } else {
      break;
    }
  }
}