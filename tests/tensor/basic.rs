//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use wasm_bindgen_test::*;

use rust_wasm_tensor::tensor::*;
use rust_wasm_tensor::assert_delta;

wasm_bindgen_test_configure!(run_in_browser);

const delta: f32 = 0.00001;

#[wasm_bindgen_test]
fn test_tensor_exp() {
    let a = Tensor::new(&vec![2,2], &vec![-1.,0.,1.,2.]);
    let expected = Tensor::new(&vec![2,2], &vec![0.367879441, 1., 2.718281828, 7.389056099]);

    assert_delta!(a.exp(), expected.clone(), Tensor::constant(&vec![2,2], delta));
}

#[wasm_bindgen_test]
fn test_tensor_log() {
    let expected = Tensor::new(&vec![2,2], &vec![-1.,0.,1.,2.]);
    let a = Tensor::new(&vec![2,2], &vec![0.367879441, 1., 2.718281828, 7.389056099]);

    assert_delta!(a.log(), expected.clone(), Tensor::constant(&vec![2,2], delta));
}

#[wasm_bindgen_test]
fn test_tensor_sqrt() {
    let a = Tensor::new(&vec![2,2], &vec![1.,4.,9.,16.]);
    let expected = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);

    assert_delta!(a.sqrt(), expected.clone(), Tensor::constant(&vec![2,2], delta));
}

#[wasm_bindgen_test]
fn test_tensor_add() {
    let a = Tensor::new(&vec![2,2], &vec![1.,4.,9.,16.]);
    let b = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let expected = Tensor::new(&vec![2,2], &vec![2.,6.,12.,20.]);

    assert_delta!(a.addition(&b), expected.clone(), Tensor::constant(&vec![2,2], delta));
}

#[wasm_bindgen_test]
fn test_tensor_subtract() {
    let a = Tensor::new(&vec![2,2], &vec![1.,4.,9.,16.]);
    let b = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let expected = Tensor::new(&vec![2,2], &vec![0.,2.,6.,12.]);

    assert_delta!(a.subtraction(&b), expected.clone(), Tensor::constant(&vec![2,2], delta));
}

#[wasm_bindgen_test]
fn test_tensor_divide() {
    let a = Tensor::new(&vec![2,3], &vec![1.,4.,9.,16.,21.,28.]);
    let b = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,7.,7.]);
    let expected = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,3.,4.]);

    assert_delta!(a.divide(&b), expected.clone(), Tensor::constant(&vec![2,3], delta));
}

#[wasm_bindgen_test]
fn test_tensor_multiply() {
    let a = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let b = Tensor::new(&vec![2,2], &vec![5.,6.,7.,8.]);
    let expected = Tensor::new(&vec![2,2], &vec![5.,12.,21.,32.]);

    assert_delta!(a.divide(&b), expected.clone(), Tensor::constant(&vec![2,2], delta));
}

#[wasm_bindgen_test]
fn test_tensor_matmul() {
    let a = Tensor::new(&vec![2,2], &vec![1.,2.,3.,4.]);
    let b = Tensor::new(&vec![2,2], &vec![5.,6.,7.,8.]);
    let expected = Tensor::new(&vec![2,2], &vec![19.,22.,43.,50.]);

    assert_delta!(a.matmul(&b), expected.clone(), Tensor::constant(&vec![2,2], delta));
}

#[wasm_bindgen_test]
fn test_tensor_matmul_dot_product() {
    let a = Tensor::new(&vec![1,4], &vec![1.,2.,3.,4.]);
    let b = Tensor::new(&vec![4,1], &vec![5.,6.,7.,8.]);
    let expected = Tensor::new(&vec![1,1], &vec![5.+12.+21.+32.]);

    assert_delta!(a.matmul(&b), expected.clone(), Tensor::constant(&vec![1,1], delta));
}