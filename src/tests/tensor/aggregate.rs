use crate::assert_delta;
use crate::ops::aggregate::*;
use crate::tensor::*;
use crate::utils::uint32_array;

const DELTA: f32 = 0.00001;

#[test]
fn test_tensor_sum() {
    let a: Tensor<f32> = Tensor::new_from_shape(&vec![2, 2], &vec![1., 2., 3., 4.]);
    let expected = Tensor::new_from_shape(&vec![1], &vec![10.]);

    assert!(a._sum(&vec![0, 1], false).compare(&expected, DELTA));
}

#[test]
fn test_tensor_sum_column_wise() {
    let a = Tensor::new_from_shape(&vec![2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let expected = Tensor::new_from_shape(&vec![3], &vec![5., 7., 9.]);

    let result = a._sum(&vec![0], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_sum_row_wise() {
    let a = Tensor::new_from_shape(&vec![2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let expected = Tensor::new_from_shape(&vec![2], &vec![6., 15.]);

    let result = a._sum(&vec![1], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_sum_multiple_axes() {
    let a = Tensor::new_from_shape(
        &vec![2, 3, 4],
        &vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ],
    );
    let expected1 = Tensor::new_from_shape(&vec![2], &vec![78., 222.]);
    let expected2 = Tensor::new_from_shape(&vec![3], &vec![68., 100., 132.]);
    let expected3 = Tensor::new_from_shape(&vec![4], &vec![66., 72., 78., 84.]);

    assert!(a._sum(&vec![1, 2], false).compare(&expected1, DELTA));
    assert!(a._sum(&vec![0, 2], false).compare(&expected2, DELTA));
    assert!(a._sum(&vec![0, 1], false).compare(&expected3, DELTA));
}

#[test]
fn test_tensor_product() {
    let a = Tensor::new_from_shape(&vec![2, 2], &vec![1., 2., 3., 4.]);
    let expected = Tensor::new_from_shape(&vec![1], &vec![24.]);

    let result = a._product(&vec![0, 1], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_product_column_wise() {
    let a = Tensor::new_from_shape(&vec![2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let expected = Tensor::new_from_shape(&vec![3], &vec![4., 10., 18.]);

    let result = a._product(&vec![0], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_product_row_wise() {
    let a = Tensor::new_from_shape(&vec![2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let expected = Tensor::new_from_shape(&vec![2], &vec![6., 120.]);

    let result = a._product(&vec![1], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_product_multiple_axes() {
    let a = Tensor::new_from_shape(
        &vec![2, 3, 4],
        &vec![
            1., 2., 3., 4., 5., 6., 7., 8., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1., 2., 1.5,
            2.5, 0.5, 1.5, 3.5, 4.5,
        ],
    );
    let expected2 = Tensor::new_from_shape(
        &vec![3],
        &vec![4.0320000648498535, 12600., 0.02835000306367874],
    );

    let result = a._product(&vec![0, 2], false);

    assert!(result.compare(&expected2, DELTA));
}

#[test]
fn test_tensor_max() {
    let a = Tensor::new_from_shape(&vec![2, 2], &vec![1., 2., 3., 4.]);
    let expected = Tensor::new_from_shape(&vec![1], &vec![4.]);

    assert!(a._max(&vec![0, 1], false).compare(&expected, DELTA));
}

#[test]
fn test_tensor_max_column_wise() {
    let a = Tensor::new_from_shape(&vec![2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let expected = Tensor::new_from_shape(&vec![3], &vec![4., 5., 6.]);

    let result = a._max(&vec![0], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_max_row_wise() {
    let a = Tensor::new_from_shape(&vec![2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let expected = Tensor::new_from_shape(&vec![2], &vec![3., 6.]);

    let result = a._max(&vec![1], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_max_multiple_axes() {
    let a = Tensor::new_from_shape(
        &vec![2, 3, 4],
        &vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ],
    );
    let expected1 = Tensor::new_from_shape(&vec![2], &vec![12., 24.]);
    let expected2 = Tensor::new_from_shape(&vec![3], &vec![16., 20., 24.]);
    let expected3 = Tensor::new_from_shape(&vec![4], &vec![21., 22., 23., 24.]);

    let res1 = a._max(&vec![1, 2], false);
    let res2 = a._max(&vec![0, 2], false);
    let res3 = a._max(&vec![0, 1], false);

    assert!(res1.compare(&expected1, DELTA));
    assert!(res2.compare(&expected2, DELTA));
    assert!(res3.compare(&expected3, DELTA));
}

#[test]
fn test_tensor_min() {
    let a = Tensor::new_from_shape(&vec![2, 2], &vec![1., 2., 3., 4.]);
    let expected = Tensor::new_from_shape(&vec![1], &vec![1.]);

    assert!(a._min(&vec![0, 1], false).compare(&expected, DELTA));
}

#[test]
fn test_tensor_min_column_wise() {
    let a = Tensor::new_from_shape(&vec![2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let expected = Tensor::new_from_shape(&vec![3], &vec![1., 2., 3.]);

    let result = a._min(&vec![0], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_min_row_wise() {
    let a = Tensor::new_from_shape(&vec![2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let expected = Tensor::new_from_shape(&vec![2], &vec![1., 4.]);

    let result = a._min(&vec![1], false);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_tensor_min_multiple_axes() {
    let a = Tensor::new_from_shape(
        &vec![2, 3, 4],
        &vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ],
    );
    let expected1 = Tensor::new_from_shape(&vec![2], &vec![1., 13.]);
    let expected2 = Tensor::new_from_shape(&vec![3], &vec![1., 5., 9.]);
    let expected3 = Tensor::new_from_shape(&vec![4], &vec![1., 2., 3., 4.]);

    let res1 = a._min(&vec![1, 2], false);
    let res2 = a._min(&vec![0, 2], false);
    let res3 = a._min(&vec![0, 1], false);

    assert!(res1.compare(&expected1, DELTA));
    assert!(res2.compare(&expected2, DELTA));
    assert!(res3.compare(&expected3, DELTA));
}

#[test]
fn test_tensor_mean() {
    let a = Tensor::new_from_shape(
        &vec![2, 2, 3],
        &vec![
            0.9762700796127319,
            4.3037872314453125,
            2.055267572402954,
            0.8976636528968811,
            -1.5269039869308472,
            2.917882204055786,
            -1.248255729675293,
            7.835460186004639,
            9.273255348205566,
            -2.331169605255127,
            5.834500789642334,
            0.577898383140564,
        ],
    );
    let expected1 = Tensor::new_from_shape(&vec![1], &vec![2.4638044834136963]);

    let res1 = a._reduce_mean(&vec![0, 1, 2], false);

    assert!(res1.compare(&expected1, DELTA));
}

#[test]
fn test_tensor_max_special_case() {
    let a = Tensor::new_from_shape(
        &vec![1, 24],
        &vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ],
    );
    let expected1 = Tensor::new_from_shape(&vec![1, 1], &vec![24.]);

    let res1 = a._max(&vec![1], true);

    assert!(res1.compare(&expected1, DELTA));
}
