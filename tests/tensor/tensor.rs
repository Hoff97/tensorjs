//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use wasm_bindgen_test::*;

use rust_wasm_tensor::shape::*;
use rust_wasm_tensor::tensor::*;

use std::cmp::Ordering;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_tensor_get_rank_1() {
    let a = Tensor::new(&vec![5], &vec![1.,2.,3.,4.,5.]);

    assert_eq!(a.get(&vec![0]), 1.);
    assert_eq!(a.get(&vec![1]), 2.);
    assert_eq!(a.get(&vec![2]), 3.);
    assert_eq!(a.get(&vec![4]), 5.);
}

#[wasm_bindgen_test]
fn test_tensor_get_rank_2() {
    let a = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,5.,6.]);

    assert_eq!(a.get(&vec![0,0]), 1.);
    assert_eq!(a.get(&vec![0,1]), 2.);
    assert_eq!(a.get(&vec![1,0]), 4.);
    assert_eq!(a.get(&vec![1,1]), 5.);
}

#[wasm_bindgen_test]
fn test_tensor_get_rank_3() {
    let a = Tensor::new(&vec![2,3,4], &vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.]);

    assert_eq!(a.get(&vec![0,0,0]), 1.);
    assert_eq!(a.get(&vec![1,0,0]), 13.);
    assert_eq!(a.get(&vec![1,1,1]), 18.);
    assert_eq!(a.get(&vec![1,2,3]), 24.);
}

#[wasm_bindgen_test]
fn test_tensor_cmp_eq() {
    let a = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,5.,6.]);
    let b = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,5.,6.]);
    let c = Tensor::new(&vec![2,3], &vec![1.,2.,3.5,4.5,5.,6.]);

    assert_eq!(a,b);
    assert_ne!(a,c);
    assert_ne!(b,c);
}

#[wasm_bindgen_test]
fn test_tensor_cmp_ordering_rank_0() {
    let a = Tensor::new(&vec![], &vec![]);
    let b = Tensor::new(&vec![], &vec![]);

    assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
}

#[wasm_bindgen_test]
fn test_tensor_cmp_ordering_rank_1() {
    let a = Tensor::new(&vec![1], &vec![1.]);
    let b = Tensor::new(&vec![1], &vec![1.]);
    let c = Tensor::new(&vec![1], &vec![2.]);
    let d = Tensor::new(&vec![1], &vec![0.]);

    assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
    assert_eq!(a.partial_cmp(&c), Some(Ordering::Less));
    assert_eq!(a.partial_cmp(&d), Some(Ordering::Greater));
}

#[wasm_bindgen_test]
fn test_tensor_cmp_ordering() {
    let a = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,5.,6.]);
    let b = Tensor::new(&vec![2,3], &vec![1.,2.,3.,4.,5.,6.]);

    let c = Tensor::new(&vec![2,3], &vec![2.,3.,4.,5.,6.,7.]);
    let d = Tensor::new(&vec![2,3], &vec![0.,1.,2.,3.,4.,5.]);

    let e = Tensor::new(&vec![2,3], &vec![0.,1.,2.,3.,4.,7.]);

    assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
    assert_eq!(a.partial_cmp(&c), Some(Ordering::Less));
    assert_eq!(a.partial_cmp(&d), Some(Ordering::Greater));
    assert_eq!(a.partial_cmp(&e), None);
}