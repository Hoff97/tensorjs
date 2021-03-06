use crate::assert_delta;
use crate::tensor::*;
use crate::utils::uint32_array;

const DELTA: f32 = 0.00001;

#[test]
fn test_tensor_conv() {
    let strides: Vec<usize> = vec![1; 2];
    let pads: Vec<usize> = vec![0; 4];
    let dilations: Vec<usize> = vec![1; 2];

    let x = Tensor::new_from_shape(&vec![1, 1, 2, 2], &vec![1., 2., 3., 4.]);
    let w = Tensor::new_from_shape(&vec![1, 1, 2, 2], &vec![1., 1., 1., 1.]);
    let b = Tensor::new_from_shape(&vec![1], &vec![5.]);

    let expected1 = Tensor::new_from_shape(&vec![1, 1, 1, 1], &vec![10.]);
    let expected2 = Tensor::new_from_shape(&vec![1, 1, 1, 1], &vec![15.]);

    let result1 = x._conv(&w, None, &dilations, 1, &pads, &strides, 0);
    let result2 = x._conv(&w, Some(&b), &dilations, 1, &pads, &strides, 0);

    assert!(result1.compare(&expected1, DELTA));
    assert!(result2.compare(&expected2, DELTA));
}

#[test]
fn test_tensor_grouped() {
    let strides: Vec<usize> = vec![1; 2];
    let pads: Vec<usize> = vec![0; 4];
    let dilations: Vec<usize> = vec![1; 2];

    let x = Tensor::new_from_shape(
        &vec![1, 2, 3, 3],
        &vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
        ],
    );
    let w = Tensor::new_from_shape(&vec![2, 1, 1, 1], &vec![1.0, 2.0]);

    let expected = Tensor::new_from_shape(
        &vec![1, 2, 3, 3],
        &vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 18., 20., 22., 24., 26., 28., 30., 32., 34.,
        ],
    );

    let result = x._conv(&w, None, &dilations, 2, &pads, &strides, 0);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_conv_1d() {
    let strides: Vec<usize> = vec![1; 1];
    let pads: Vec<usize> = vec![0; 2];
    let dilations: Vec<usize> = vec![1; 1];

    let x = Tensor::new_from_shape(&vec![1, 1, 6], &vec![1., 2., 3., 4., 5., 6.]);
    let w = Tensor::new_from_shape(&vec![1, 1, 3], &vec![1.0, 2.0, 3.]);

    let expected = Tensor::new_from_shape(&vec![1, 1, 4], &vec![14., 20., 26., 32.]);

    let result = x._conv(&w, None, &dilations, 1, &pads, &strides, 1);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_conv_1d_relu() {
    let strides: Vec<usize> = vec![1; 1];
    let pads: Vec<usize> = vec![0; 2];
    let dilations: Vec<usize> = vec![1; 1];

    let x = Tensor::new_from_shape(&vec![1, 1, 6], &vec![1., 2., 3., -4., 5., 6.]);
    let w = Tensor::new_from_shape(&vec![1, 1, 3], &vec![1.0, 2.0, 3.]);

    let expected = Tensor::new_from_shape(&vec![1, 1, 4], &vec![14., 0.0, 10.0, 24.0]);

    let result = x._conv(&w, None, &dilations, 1, &pads, &strides, 1);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_conv_3d() {
    let strides: Vec<usize> = vec![1; 3];
    let pads: Vec<usize> = vec![0; 6];
    let dilations: Vec<usize> = vec![1; 3];

    let t: Vec<u32> = (1..=27).collect();
    let y: Vec<f32> = t.iter().map(|&x| x as f32).collect();

    let x = Tensor::new_from_shape(&vec![1, 1, 3, 3, 3], &y);
    let w = Tensor::new_from_shape(&vec![1, 1, 2, 2, 2], &vec![1., 2., 3., 4., 5., 6., 7., 8.]);

    let expected = Tensor::new_from_shape(
        &vec![1, 1, 2, 2, 2],
        &vec![356., 392., 464., 500., 680., 716., 788., 824.],
    );

    let result = x._conv(&w, None, &dilations, 1, &pads, &strides, 0);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_conv_dilated() {
    let strides: Vec<usize> = vec![1; 2];
    let pads: Vec<usize> = vec![0; 4];
    let dilations: Vec<usize> = vec![2; 2];

    let t: Vec<u32> = (1..=16).collect();
    let y: Vec<f32> = t.iter().map(|&x| x as f32).collect();

    let x = Tensor::new_from_shape(&vec![1, 1, 4, 4], &y);
    let w = Tensor::new_from_shape(&vec![1, 1, 2, 2], &vec![1., 2., 3., 4.]);

    let expected = Tensor::new_from_shape(&vec![1, 1, 2, 2], &vec![78., 88., 118., 128.]);

    let result = x._conv(&w, None, &dilations, 1, &pads, &strides, 0);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_conv_strided() {
    let strides: Vec<usize> = vec![2; 2];
    let pads: Vec<usize> = vec![0; 4];
    let dilations: Vec<usize> = vec![1; 2];

    let t: Vec<u32> = (1..=16).collect();
    let y: Vec<f32> = t.iter().map(|&x| x as f32).collect();

    let x = Tensor::new_from_shape(&vec![1, 1, 4, 4], &y);
    let w = Tensor::new_from_shape(&vec![1, 1, 2, 2], &vec![1., 2., 3., 4.]);

    let expected = Tensor::new_from_shape(&vec![1, 1, 2, 2], &vec![44., 64., 124., 144.]);

    let result = x._conv(&w, None, &dilations, 1, &pads, &strides, 0);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_conv_padded() {
    let strides: Vec<usize> = vec![1; 2];
    let pads: Vec<usize> = vec![1; 4];
    let dilations: Vec<usize> = vec![1; 2];

    let x = Tensor::new_from_shape(&vec![1, 1, 2, 2], &vec![1., 2., 3., 4.]);
    let w = Tensor::new_from_shape(&vec![1, 1, 2, 2], &vec![1., 2., 3., 4.]);

    let expected = Tensor::new_from_shape(
        &vec![1, 1, 3, 3],
        &vec![4., 11., 6., 14., 30., 14., 6., 11., 4.],
    );

    let result = x._conv(&w, None, &dilations, 1, &pads, &strides, 0);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_conv_multiple_filters() {
    let strides: Vec<usize> = vec![1; 1];
    let pads: Vec<usize> = vec![0; 2];
    let dilations: Vec<usize> = vec![1; 1];

    let x = Tensor::new_from_shape(&vec![1, 2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let w = Tensor::new_from_shape(
        &vec![3, 2, 2],
        &vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
    );

    let expected = Tensor::new_from_shape(
        &vec![1, 3, 2],
        &vec![
            5. + 12. + 20.,
            2. + 6. + 15. + 24.,
            5. + 12. + 28. + 40.,
            10. + 18. + 35. + 48.,
            9. + 20. + 44. + 60.,
            175.,
        ],
    );

    let result = x._conv(&w, None, &dilations, 1, &pads, &strides, 0);

    assert!(result.compare(&expected, DELTA));
}

#[test]
fn test_conv_transpose_no_pad() {
    let strides: Vec<usize> = vec![1; 1];
    let pads: Vec<usize> = vec![0; 2];
    let dilations: Vec<usize> = vec![1; 1];

    let x = Tensor::new_from_shape(&vec![1, 2, 3], &vec![1., 2., 3., 4., 5., 6.]);
    let w = Tensor::new_from_shape(&vec![1, 2, 2], &vec![1., 2., 3., 4.]);

    let result = x._conv_transpose(&w, &dilations, 1, &pads, &strides);
}

#[test]
fn test_conv_back() {
    let strides: Vec<usize> = vec![1; 2];
    let pads: Vec<usize> = vec![1; 4];
    let dilations: Vec<usize> = vec![1; 2];

    let x = Tensor::new_from_shape(&vec![1, 1, 3, 3], &vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    let w = Tensor::new_from_shape(
        &vec![1, 1, 4, 4],
        &vec![
            1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
        ],
    );

    let result = x._conv(&w, None, &dilations, 1, &pads, &strides, 0);
    let result2 = x._conv(&w, None, &dilations, 1, &pads, &strides, 0);
}
//[1, 1, 3, 3], [1, 1, 4, 4], [1, 1], [1, 1, 1, 1], [1, 1]
