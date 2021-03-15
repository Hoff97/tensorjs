use crate::assert_delta;
use crate::tensor::*;
use crate::utils::uint32_array;

extern crate test;

use test::Bencher;

#[bench]
fn bench_add(b: &mut Bencher) {
    let x = Tensor::new_from_shape(&vec![2, 2], &vec![1., 2., 3., 4.]);
    let y = Tensor::new_from_shape(&vec![2, 2], &vec![5., 6., 7., 8.]);
    b.iter(|| x.addition(&y, 1., 1.));
}
