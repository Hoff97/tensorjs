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
fn test_tensor_gemm() {
    let a = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let b = Tensor::new(&vec![2,2], &vec![5.,6.,7.,8.]);
    let expected = Tensor::new(&vec![2,2], &vec![19., 22., 43., 50.]);

    assert!(a._gemm(&b, false, false, 1.0, None, 1.0).compare(&expected, DELTA));
}

#[test]
fn test_tensor_gemm_a_b_transposed_and_c() {
    let a = Tensor::new(&vec![2,3,2], &vec![1.,4.,2.,5.,3.,6.,7.,10.,8.,11.,9.,12.]);
    let b = Tensor::new(&vec![2,2,3], &vec![7.,9.,11.,8.,10.,12.,13.,15.,17.,14.,16.,18.]);

    let expected1 = Tensor::new(&vec![2,2,2], &vec![29. + 1.,32. + 1.,69.5 + 1.,77. + 1.,182. + 1.,194. + 1.,249.5 + 1.,266. + 1.]);
    let expected2 = Tensor::new(&vec![2,2,2], &vec![29. + 1.,32. + 2.,69.5 + 1.,77. + 2.,182. + 1.,194. + 2.,249.5 + 1.,266. + 2.]);
    let expected3 = Tensor::new(&vec![2,2,2], &vec![29. + 1.,32. + 2.,69.5 + 3.,77. + 4.,182. + 1.,194. + 2.,249.5 + 3.,266. + 4.]);
    let expected4 = Tensor::new(&vec![2,2,2], &vec![29. + 1.,32. + 2.,69.5 + 3.,77. + 4.,182. + 5.,194. + 6.,249.5 + 7.,266. + 8.]);

    let c1 = Tensor::new(&vec![1,1,1], &vec![1.]);
    let c2 = Tensor::new(&vec![1,1,2], &vec![1.,2.]);
    let c3 = Tensor::new(&vec![1,2,2], &vec![1.,2.,3.,4.]);
    let c4 = Tensor::new(&vec![2,2,2], &vec![1.,2.,3.,4.,5.,6.,7.,8.]);
    let alpha = 0.5;

    assert!(a._gemm(&b, true, true, alpha, Some(&c1), 1.0).compare(&expected1, DELTA));
    assert!(a._gemm(&b, true, true, alpha, Some(&c2), 1.0).compare(&expected2, DELTA));
    assert!(a._gemm(&b, true, true, alpha, Some(&c3), 1.0).compare(&expected3, DELTA));
    assert!(a._gemm(&b, true, true, alpha, Some(&c4), 1.0).compare(&expected4, DELTA));
}