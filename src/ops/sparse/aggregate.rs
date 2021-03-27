use crate::shape::*;
use crate::tensor::*;
use js_sys::Uint32Array;
use num_traits::zero;
use num_traits::FromPrimitive;
use num_traits::Num;

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: FromPrimitive,
{
    pub fn aggregate_sparse<F, F2, F3>(
        &self,
        shape: &Vec<usize>,
        indices: &Tensor<u32>,
        axes: &Vec<usize>,
        keep_dims: bool,
        op: F,
        init: bool,
        init_op: F2,
        post: bool,
        post_op: F3,
    ) -> Tensor<DType>
    where
        F: Fn(DType, DType) -> DType,
        F2: Fn(DType) -> DType,
        F3: Fn(DType, usize) -> DType,
    {
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
        let mut values: Vec<DType> = vec![zero(); result_size];
        let mut count = vec![0; result_size];

        let mut sparse_ix = vec![0; s];

        for i in 0..nnz {
            for j in 0..s {
                sparse_ix[j] = indices.get_ix(i * s + j);
            }

            let mut dense_ix = vec![0; d];
            for j in 0..dense_size {
                let v = self.get_values()[i * dense_size + j];

                let mut out_pos = 0;
                for k in 0..result_rank {
                    let i = res_ix_map[k];
                    if i < s {
                        out_pos += sparse_ix[i] as usize * result_strides[k];
                    } else {
                        out_pos += dense_ix[i - s] * result_strides[k];
                    }
                }

                if init && count[out_pos] == 0 {
                    values[out_pos] = init_op(v);
                } else {
                    values[out_pos] = op(values[out_pos], v);
                }

                count[out_pos] += 1;

                increment_index_slice(&mut dense_ix, &self.get_sh()[1..]);
            }
        }

        if post {
            for i in 0..result_size {
                values[i] = post_op(values[i], count[i]);
            }
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }

    pub fn _sum_sparse(
        &self,
        shape: &Vec<usize>,
        indices: &Tensor<u32>,
        axes: &Vec<usize>,
        keep_dims: bool,
    ) -> Tensor<DType> {
        self.aggregate_sparse(
            shape,
            indices,
            axes,
            keep_dims,
            |a: DType, b: DType| a + b,
            false,
            |a: DType| a,
            false,
            |a: DType, b: usize| a,
        )
    }

    pub fn sum_sparse(
        &self,
        shape: Uint32Array,
        indices: &Tensor<u32>,
        axes: Uint32Array,
        keep_dims: bool,
    ) -> Tensor<DType> {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..shape.length() {
            _shape[i as usize] = shape.get_index(i as u32) as usize;
        }

        let mut _axes: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            _axes[i as usize] = axes.get_index(i as u32) as usize;
        }

        self._sum_sparse(&_shape, indices, &_axes, keep_dims)
    }

    pub fn _sum_square_sparse(
        &self,
        shape: &Vec<usize>,
        indices: &Tensor<u32>,
        axes: &Vec<usize>,
        keep_dims: bool,
    ) -> Tensor<DType> {
        self.aggregate_sparse(
            shape,
            indices,
            axes,
            keep_dims,
            |a: DType, b: DType| a + b * b,
            true,
            |a: DType| a * a,
            false,
            |a: DType, b: usize| a,
        )
    }

    pub fn sum_square_sparse(
        &self,
        shape: Uint32Array,
        indices: &Tensor<u32>,
        axes: Uint32Array,
        keep_dims: bool,
    ) -> Tensor<DType> {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..shape.length() {
            _shape[i as usize] = shape.get_index(i as u32) as usize;
        }

        let mut _axes: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            _axes[i as usize] = axes.get_index(i as u32) as usize;
        }

        self._sum_square_sparse(&_shape, indices, &_axes, keep_dims)
    }

    pub fn _reduce_mean_sparse(
        &self,
        shape: &Vec<usize>,
        indices: &Tensor<u32>,
        axes: &Vec<usize>,
        keep_dims: bool,
    ) -> Tensor<DType> {
        self.aggregate_sparse(
            shape,
            indices,
            axes,
            keep_dims,
            |a: DType, b: DType| a + b,
            false,
            |a: DType| a,
            true,
            |a: DType, b: usize| a / DType::from_usize(b).expect("Error in sparse mean: Data type can not represent number of items contained in dimension"),
        )
    }

    pub fn reduce_mean_sparse(
        &self,
        shape: Uint32Array,
        indices: &Tensor<u32>,
        axes: Uint32Array,
        keep_dims: bool,
    ) -> Tensor<DType> {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..shape.length() {
            _shape[i as usize] = shape.get_index(i as u32) as usize;
        }

        let mut _axes: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            _axes[i as usize] = axes.get_index(i as u32) as usize;
        }

        self._reduce_mean_sparse(&_shape, indices, &_axes, keep_dims)
    }
}
