use js_sys::Uint32Array;

pub fn get_size(shape: &Vec<usize>) -> usize {
    if shape.len() == 0 {
        return 0;
    }

    let mut size: usize = 1;
    for sh in shape {
        size *= sh;
    }
    return size;
}

pub fn get_size_from(shape: &Vec<usize>, start_ix: usize) -> usize {
    if shape.len() == 0 {
        return 0;
    }

    let mut size: usize = 1;
    for i in start_ix..shape.len() {
        size *= shape[i];
    }
    return size;
}

pub fn get_size_from_to(shape: &Vec<usize>, start_ix: usize, end_ix: usize) -> usize {
    if shape.len() == 0 {
        return 0;
    }

    let mut size: usize = 1;
    for i in start_ix..end_ix {
        size *= shape[i];
    }
    return size;
}

pub fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let rank = shape.len();

    if rank == 0 {
        return vec![];
    }
    if rank == 1 {
        if shape[0] == 1 {
            return vec![0];
        }
        return vec![1];
    }

    let mut res: Vec<usize> = vec![1; rank];
    if shape[rank - 1] == 1 {
        res[rank - 1] = 0;
    } else {
        res[rank - 1] = 1;
    }
    let mut last_stride = 1;

    for i in (0..(shape.len()-1)).rev() {
        last_stride = last_stride * shape[i+1];
        if shape[i] == 1 {
            res[i] = 0;
        } else {
            res[i] = last_stride;
        }
    }

    return res;
}

pub fn compute_strides_no_zero(shape: &Vec<usize>) -> Vec<usize> {
    let rank = shape.len();

    if rank == 0 {
        return vec![];
    }
    if rank == 1 {
        return vec![1];
    }

    let mut res: Vec<usize> = vec![1; rank];

    let mut last_stride = 1;

    for i in (0..(shape.len()-1)).rev() {
        last_stride = last_stride * shape[i+1];
        res[i] = last_stride;
    }

    return res;
}

pub fn compute_strides_uint32(shape: &Uint32Array) -> Vec<usize> {
    let rank = shape.length();

    if rank == 0 {
        return vec![];
    }
    if rank == 1 {
        if shape.get_index(0) == 0 {
            return vec![0];
        }
        return vec![1];
    }

    let mut res: Vec<usize> = vec![1; rank as usize];
    if shape.get_index(rank - 1) == 1 {
        res[rank as usize - 1] = 0;
    } else {
        res[rank as usize - 1] = 1;
    }
    let mut last_stride = 1;

    for i in (0..(shape.length()-1)).rev() {
        last_stride = last_stride * (shape.get_index(i + 1) as usize);
        if shape.get_index(i) == 1 {
            res[i as usize] = 0;
        } else {
            res[i as usize] = last_stride;
        }
    }

    return res;
}

pub fn compare_shapes(a: &Vec<usize>, b: &Vec<usize>) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        if a[i] != b[i] {
            return false;
        }
    }

    return true;
}

pub fn index_to_pos(index: &Vec<usize>, strides: &Vec<usize>) -> usize {
    let mut ix: usize = 0;
    for i in 0..index.len() {
        ix += index[i] * strides[i];
    }
    return ix;
}

pub fn pos_to_index(pos: usize, strides: &Vec<usize>) -> Vec<usize> {
    let mut ix = vec![0; strides.len()];

    let mut pos_c = pos;

    for i in 0..strides.len() {
        ix[i] = pos_c / strides[i];
        pos_c %= strides[i];
    }
    return ix;
}

pub fn increment_index(index: &mut Vec<usize>, shape: &Vec<usize>) {
    for i in (0..index.len()).rev() {
        index[i] += 1;
        if index[i] >= shape[i] {
            index[i] = 0;
        } else {
            return;
        }
    }
}

pub fn decrement_index(index: &mut Vec<usize>, shape: &Vec<usize>) {
    for i in (0..index.len()).rev() {
        if index[i] == 0 {
            index[i] = shape[i] - 1;
        } else {
            index[i] -= 1;
            return;
        }
    }
}