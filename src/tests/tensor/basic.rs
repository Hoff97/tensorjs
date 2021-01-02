use crate::tensor::*;
use crate::assert_delta;
use crate::utils::uint32_array;

const DELTA: f32 = 0.00001;

#[test]
fn test_tensor_exp() {
    let a = Tensor::new(&vec![2,2], &vec![-1.,0.,1.,2.]);
    let expected = Tensor::new(&vec![2,2], &vec![0.367879441, 1.0, 2.718281828, 7.389056099]);

    assert!(a.exp().compare(&expected, DELTA));
}

#[test]
fn test_tensor_log() {
    let expected = Tensor::new(&vec![2,2], &vec![-1.,0.,1.,2.]);
    let a = Tensor::new(&vec![2,2], &vec![0.367879441, 1., 2.718281828, 7.389056099]);

    assert!(a.log().compare(&expected, DELTA));
}

#[test]
fn test_tensor_sqrt() {
    let a = Tensor::new(&vec![2,2], &vec![1.,4.,9.,16.]);
    let expected = Tensor::new(&vec![2,2], &vec![1.0,2.,3.,4.]);

    assert!(a.sqrt().compare(&expected, DELTA));
}

#[test]
fn test_tensor_add() {
    let a = Tensor::new(&vec![2,2], &vec![1.,4.,9.,16.]);
    let b = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let expected = Tensor::new(&vec![2,2], &vec![2.,6.,12.,20.]);

    assert!(a.addition(&b).compare(&expected, DELTA));
}

#[test]
fn test_tensor_subtract() {
    let a = Tensor::new(&vec![2,2], &vec![1.,4.,9.,16.]);
    let b = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let expected = Tensor::new(&vec![2,2], &vec![0.,2.,6.,12.]);

    assert!(a.subtraction(&b).compare(&expected, DELTA));
}

#[test]
fn test_tensor_divide() {
    let a = Tensor::new(&vec![2,3], &vec![1.,4.,9.,16.,21.,28.]);
    let b = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,7.,7.]);
    let expected = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,3.,4.]);

    assert!(a.divide(&b).compare(&expected, DELTA));
}

#[test]
fn test_tensor_multiply() {
    let a = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let b = Tensor::new(&vec![2,2], &vec![5.,6.,7.,8.]);
    let expected = Tensor::new(&vec![2,2], &vec![5.,12.,21.,32.]);

    assert!(a.multiply(&b).compare(&expected, DELTA));
}

#[test]
fn test_tensor_matmul() {
    let a = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let b = Tensor::new(&vec![2,2], &vec![5.,6.,7.,8.]);
    let expected = Tensor::new(&vec![2,2], &vec![19.,22.,43.,50.]);

    assert!(a.matmul(&b).compare(&expected, DELTA));
}

#[test]
fn test_tensor_matmul_dot_product() {
    let a = Tensor::new(&vec![1,4], &vec![1.,2.,3.,4.]);
    let b = Tensor::new(&vec![4,1], &vec![5.,6.,7.,8.]);
    let expected = Tensor::new(&vec![1,1], &vec![5.+12.+21.+32.]);

    assert!(a.matmul(&b).compare(&expected, DELTA));
}

#[test]
fn test_tensor_sum() {
    let a = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let expected = Tensor::new(&vec![1], &vec![10.]);

    assert!(a._sum(&vec![0, 1]).compare(&expected, DELTA));
}

#[test]
fn test_tensor_sum_column_wise() {
    let a = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,5.,6.]);
    let expected = Tensor::new(&vec![3], &vec![5.,7.,9.]);

    let result = a._sum(&vec![0]);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_sum_row_wise() {
    let a = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,5.,6.]);
    let expected = Tensor::new(&vec![2], &vec![6.,15.]);

    let result = a._sum(&vec![1]);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_sum_multiple_axes() {
    let a = Tensor::new(&vec![2,3,4], &vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.]);
    let expected1 = Tensor::new(&vec![2], &vec![78.,222.]);
    let expected2 = Tensor::new(&vec![3], &vec![68., 100., 132.]);
    let expected3 = Tensor::new(&vec![4], &vec![66., 72., 78., 84.]);

    assert!(a._sum(&vec![1,2]).compare(&expected1, DELTA));
    assert!(a._sum(&vec![0,2]).compare(&expected2, DELTA));
    assert!(a._sum(&vec![0,1]).compare(&expected3, DELTA));
}