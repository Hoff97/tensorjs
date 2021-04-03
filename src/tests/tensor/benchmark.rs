use crate::assert_delta;
use crate::shape::*;
use crate::tensor::*;
use crate::utils::uint32_array;
use rand::Rng;

extern crate test;

use test::Bencher;

fn random_tensor(shape: &Vec<usize>) -> Tensor<f32> {
    let mut rng = rand::thread_rng();

    let size = get_size(shape);
    let strides = compute_strides(shape);

    let mut values = vec![0.0; size];
    for i in 0..size {
        values[i] = rng.gen();
    }

    Tensor::new_from_shape(shape, &values)
}

#[bench]
fn bench_add(b: &mut Bencher) {
    let x = random_tensor(&vec![100, 100]);
    let y = random_tensor(&vec![100, 100]);
    b.iter(|| x.addition(&y, 1., 1.));
}

#[bench]
fn bench_matmul(b: &mut Bencher) {
    let x = random_tensor(&vec![100, 100]);
    let y = random_tensor(&vec![100, 100]);
    b.iter(|| x.matmul(&y));
}

#[bench]
fn bench_gemm(b: &mut Bencher) {
    let x = random_tensor(&vec![256, 128]);
    let y = random_tensor(&vec![128, 256]);
    b.iter(|| x._gemm(&y, false, false, 1.0, None, 1.0));
}

#[bench]
fn bench_gemm_with_c(b: &mut Bencher) {
    let x = random_tensor(&vec![256, 128]);
    let y = random_tensor(&vec![128, 256]);
    let c = random_tensor(&vec![1, 256]);
    b.iter(|| x._gemm(&y, false, false, 1.0, Some(&c), 1.0));
}
