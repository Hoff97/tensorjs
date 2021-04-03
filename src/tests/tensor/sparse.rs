use crate::assert_delta;
use crate::tensor::*;
use crate::utils::uint32_array;

const DELTA: u32 = 1;

#[test]
fn test_tensor_sparse_reshape() {
    let a: Tensor<u32> = Tensor::new_from_shape(&vec![1, 1], &vec![0]);
    let expected = Tensor::new_from_shape(&vec![2, 1], &vec![0, 1]);

    let result = a._reshape_sparse_indices(&vec![2], &vec![4]);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_sparse_reshape_only_dense() {
    let a: Tensor<u32> = Tensor::new_from_shape(&vec![1, 1], &vec![0]);
    let expected = Tensor::new_from_shape(&vec![1, 1], &vec![0]);

    let result = a._reshape_sparse_indices(&vec![2], &vec![2, 4]);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_sparse_reshape_only_sparse() {
    let a: Tensor<u32> = Tensor::new_from_shape(&vec![2, 2], &vec![0, 0, 1, 1]);
    let expected = Tensor::new_from_shape(&vec![2, 1], &vec![0, 3]);

    let result = a._reshape_sparse_indices(&vec![2, 2], &vec![4, 1]);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_sparse_reshape_all() {
    let a: Tensor<u32> = Tensor::new_from_shape(&vec![3, 2], &vec![0, 0, 0, 2, 1, 1]);
    let expected = Tensor::new_from_shape(&vec![6, 1], &vec![0, 1, 4, 5, 8, 9]);

    let result = a._reshape_sparse_indices(&vec![2, 3], &vec![12, 2]);

    assert!(result.compare(&expected, DELTA));
}
