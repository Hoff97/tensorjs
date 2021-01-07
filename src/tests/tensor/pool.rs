use crate::tensor::*;
use crate::assert_delta;
use crate::utils::uint32_array;

const DELTA: f32 = 0.00001;

#[test]
fn test_tensor_sum() {
    let a = Tensor::new(&vec![1,1,2,2], &vec![1.,2.,3.,4.]);
    let expected = Tensor::new(&vec![1,1,1,1], &vec![10./4.0]);

    let result = a._average_pool(&vec![2,2], &vec![0; 4], &vec![1; 2], false);

    assert!(result.compare(&expected, DELTA));
}
