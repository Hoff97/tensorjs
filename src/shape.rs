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

pub fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let rank = shape.len();

    if rank == 0 {
        return vec![];
    }
    if rank == 1 {
        return vec![1];
    }

    let mut res: Vec<usize> = vec![1; rank];
    res[rank - 1] = 1;

    for i in (0..(shape.len()-1)).rev() {
        res[i] = res[i+1] * shape[i+1];
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