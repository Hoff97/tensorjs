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
