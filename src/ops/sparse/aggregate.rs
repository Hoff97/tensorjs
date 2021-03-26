use crate::shape::*;
use crate::tensor::*;
use js_sys::Uint32Array;
use num_traits::zero;
use num_traits::Num;
use std::collections::HashMap;

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
{
    pub fn aggregate_sparse<F>() {}

    pub fn sum(
        &self,
        shape: &Vec<usize>,
        indices: &Tensor<u32>,
        axes: &Vec<usize>,
        keep_dims: bool,
    ) -> Tensor<u32> {
        let nnz = self.get_dim_size(0);
        let s = indices.get_dim_size(1);
        let d = self.rank() - 1;

        let mut result_rank = shape.len() - axes.len() as usize;

        if keep_dims {
            result_rank = shape.len();
        }

        let mut result_shape = vec![0; result_rank];
        let mut result_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        let mut res_ix_map = vec![0; result_rank];
        let mut dense_size = 1;
        for i in 0..shape.len() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
                if keep_dims {
                    result_shape[result_ix] = 1;
                    res_ix_map[result_ix] = i;
                    result_ix += 1;
                }
            } else {
                result_shape[result_ix] = shape[i];
                res_ix_map[result_ix] = i;
                result_ix += 1;
                result_size *= shape[i];
            }

            if i >= s {
                dense_size *= shape[i];
            }
        }

        let result_strides = compute_strides(&result_shape);
        let mut values = vec![zero(); result_size];
        let mut initialized = vec![false; result_size];

        for i in 0..nnz {
            let mut ix = vec![0; s + d];
            let mut sparse_pos = 0;
            for j in 0..s {
                sparse_ix[j] = indices.get_ix(i * s + j);
            }

            let mut dense_ix = vec![0; d];
            for j in 0..dense_size {
                increment_index(&mut dense_ix, self.get_sh()[1..]);
            }
        }
    }
}
