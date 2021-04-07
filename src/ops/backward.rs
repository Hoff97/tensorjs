use crate::ops::aggregate::pool_result;
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
    DType: PartialOrd,
    DType: FromPrimitive,
{
    pub fn _max_backward(&self, value: &Tensor<DType>, axes: &Vec<usize>) -> Tensor<DType> {
        let result_shape = value.get_sh().to_owned();
        let result_strides = value.get_strides().to_owned();
        let result_size = value.size;

        let mut result_values = vec![zero(); result_size];

        let arg_max = value._arg_max(axes, false);

        let (pool_shape, ix_map) = pool_result(value.get_sh(), axes, false);

        let mut result_ix = vec![0; value.rank()];
        let mut grad_ix: Vec<usize> = vec![0; pool_shape.len()];

        for i in 0..self.size {
            let mut result_pos = 0;
            for j in 0..pool_shape.len() {
                result_ix[ix_map[j]] = grad_ix[j];
                result_pos += result_strides[ix_map[j]] * grad_ix[j];
            }
            for j in 0..axes.len() {
                result_ix[axes[j]] = arg_max.get_ix(i * axes.len() + j) as usize;
                result_pos +=
                    result_strides[axes[j]] * (arg_max.get_ix(i * axes.len() + j) as usize);
            }

            result_values[result_pos] = self.get_ix(i);

            increment_index(&mut grad_ix, &pool_shape);
        }

        Tensor::new(result_shape, result_strides, result_size, result_values)
    }

    pub fn max_backward(&self, value: &Tensor<DType>, axes: Uint32Array) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._max_backward(value, &ax);
    }

    pub fn _min_backward(&self, value: &Tensor<DType>, axes: &Vec<usize>) -> Tensor<DType> {
        let result_shape = value.get_sh().to_owned();
        let result_strides = value.get_strides().to_owned();
        let result_size = value.size;

        let mut result_values = vec![zero(); result_size];

        let arg_min = value._arg_min(axes, false);

        let (pool_shape, ix_map) = pool_result(value.get_sh(), axes, false);

        let mut result_ix = vec![0; value.rank()];
        let mut grad_ix: Vec<usize> = vec![0; pool_shape.len()];

        for i in 0..self.size {
            let mut result_pos = 0;
            for j in 0..pool_shape.len() {
                result_ix[ix_map[j]] = grad_ix[j];
                result_pos += result_strides[ix_map[j]] * grad_ix[j];
            }
            for j in 0..axes.len() {
                result_ix[axes[j]] = arg_min.get_ix(i * axes.len() + j) as usize;
                result_pos +=
                    result_strides[axes[j]] * (arg_min.get_ix(i * axes.len() + j) as usize);
            }

            result_values[result_pos] = self.get_ix(i);

            increment_index(&mut grad_ix, &pool_shape);
        }

        Tensor::new(result_shape, result_strides, result_size, result_values)
    }

    pub fn min_backward(&self, value: &Tensor<DType>, axes: Uint32Array) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._min_backward(value, &ax);
    }
}
