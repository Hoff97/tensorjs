use crate::tensor::*;
use num_traits::zero;
use num_traits::Num;

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
{
    pub fn matmul_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        m: usize,
    ) -> Tensor<DType> {
        let o = b.get_dim_size(1);

        let result_shape = vec![m, o];
        let result_strides = vec![o, 1];
        let result_size = m * o;
        let mut result_values = vec![zero(); result_size];

        for ix in 0..indices.get_dim_size(0) {
            let i = indices.get_ix(ix * 2) as usize;
            let j = indices.get_ix(ix * 2 + 1) as usize;

            for k in 0..o {
                result_values[i * o + k] =
                    result_values[i * o + k] + self.get_ix(ix) * b.get_ix(j * o + k);
            }
        }

        Tensor::new(result_shape, result_strides, result_size, result_values)
    }
}
