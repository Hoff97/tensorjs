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
    let x = Tensor::new_from_shape(&vec![2, 2], &vec![1., 2., 3., 4.]);
    let y = Tensor::new_from_shape(&vec![2, 2], &vec![5., 6., 7., 8.]);
    b.iter(|| x.addition(&y, 1., 1.));
}

#[bench]
fn bench_matmul(b: &mut Bencher) {
    let x = random_tensor(&vec![100, 100]);
    let y = random_tensor(&vec![100, 100]);
    b.iter(|| x.matmul(&y));
}
