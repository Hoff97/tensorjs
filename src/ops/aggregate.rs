use crate::shape::*;
use crate::tensor::*;
use js_sys::Uint32Array;
use num_traits::zero;
use num_traits::Float;
use num_traits::FromPrimitive;
use num_traits::Num;

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: PartialOrd,
    DType: FromPrimitive,
{
    #[inline]
    pub fn pool_continuous<F, F2, F3>(
        &self,
        axes: &Vec<usize>,
        keep_dims: bool,
        op: F,
        postprocess: bool,
        post: F2,
        init: bool,
        init_func: F3,
    ) -> Tensor<DType>
    where
        F: Fn(DType, DType) -> DType,
        F2: Fn(DType) -> DType,
        F3: Fn(DType) -> DType,
    {
        let mut result_rank = self.rank() - axes.len() as usize;

        if keep_dims {
            result_rank = self.rank()
        }

        let mut result_shape = vec![0; result_rank];
        let mut result_size = 1;
        let mut sum_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        for i in 0..self.rank() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
                sum_size *= self.get_dim_size(i);
                if keep_dims {
                    result_shape[result_ix] = 1;
                    result_ix += 1;
                }
            } else {
                result_shape[result_ix] = self.get_dim_size(i);
                result_ix += 1;
                result_size *= self.get_dim_size(i);
            }
        }
        let result_strides = compute_strides(&result_shape);
        let mut values = vec![zero(); result_size];

        let self_strides = compute_strides_no_zero(self.get_sh());

        let step_size = self_strides[axes[axes.len() - 1]];
        let cont_size = step_size;

        let mut input_ix_step_size = if axes[0] > 0 {
            self_strides[axes[0] - 1]
        } else {
            self.size
        };
        if input_ix_step_size == 0 {
            input_ix_step_size = 1;
        }
        let num_input_steps = self.size / input_ix_step_size;

        let mut input_start_ix;
        let mut output_ix;
        for i in 0..num_input_steps {
            input_start_ix = i * input_ix_step_size;
            output_ix = i * cont_size;

            for j in 0..cont_size {
                let mut res = self.get_ix(input_start_ix + j);
                if init {
                    res = init_func(res);
                }
                for k in 1..sum_size {
                    res = op(self.get_ix(input_start_ix + j + k * step_size), res);
                }
                if postprocess {
                    res = post(res);
                }
                values[output_ix + j] = res;
            }
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }

    pub fn axes_continuous(&self, axes: &Vec<usize>) -> bool {
        let mut last_ax = axes[0];
        for i in 1..axes.len() {
            if axes[i] > last_ax + 1 {
                return false;
            }
            last_ax = axes[i];
        }
        return true;
    }

    #[inline]
    pub fn _pool<F, F2, F3>(
        &self,
        axes: &Vec<usize>,
        keep_dims: bool,
        op: F,
        postprocess: bool,
        post: F2,
        init: bool,
        init_func: F3,
    ) -> Tensor<DType>
    where
        F: Fn(DType, DType) -> DType,
        F2: Fn(DType) -> DType,
        F3: Fn(DType) -> DType,
    {
        let mut result_rank = self.rank() - axes.len() as usize;

        if keep_dims {
            result_rank = self.rank();
        }

        if result_rank == 0 {
            let mut value = self.get_ix(0);
            if init {
                value = init_func(value);
            }
            for i in 1..self.size {
                value = op(self.get_ix(i), value);
            }

            if postprocess {
                value = post(value);
            }

            return Tensor::new(
                if keep_dims {
                    vec![1; result_rank]
                } else {
                    vec![1; 1]
                },
                vec![1],
                1,
                vec![value],
            );
        }

        if self.axes_continuous(axes) {
            return self.pool_continuous(axes, keep_dims, op, postprocess, post, init, init_func);
        }

        let mut result_shape = vec![0; result_rank];
        let mut result_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        let mut res_ix_map = vec![0; result_rank];
        for i in 0..self.rank() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
                if keep_dims {
                    result_shape[result_ix] = 1;
                    res_ix_map[result_ix] = i;
                    result_ix += 1;
                }
            } else {
                result_shape[result_ix] = self.get_dim_size(i);
                res_ix_map[result_ix] = i;
                result_ix += 1;
                result_size *= self.get_dim_size(i);
            }
        }
        let result_strides = compute_strides(&result_shape);
        let mut values = vec![zero(); result_size];
        let mut initialized = vec![false; result_size];

        let mut input_index = vec![0; self.rank()];
        for i in 0..self.size {
            let mut res_ix = 0;
            for j in 0..result_rank {
                res_ix += result_strides[j] * input_index[res_ix_map[j]];
            }
            if !initialized[res_ix] {
                values[res_ix] = self.get_ix(i);
                if init {
                    values[res_ix] = init_func(values[res_ix]);
                }
                initialized[res_ix] = true;
            } else {
                values[res_ix] = op(self.get_ix(i), values[res_ix]);
            }

            increment_index(&mut input_index, self.get_sh());
        }

        if postprocess {
            for i in 0..result_size {
                values[i] = post(values[i]);
            }
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }

    pub fn _sum(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        return self._pool(
            axes,
            keep_dims,
            |x: DType, y: DType| x + y,
            false,
            |x: DType| x,
            false,
            |x: DType| x,
        );
    }

    pub fn _sum_square(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        return self._pool(
            axes,
            keep_dims,
            |x: DType, y: DType| (x * x) + y,
            false,
            |x: DType| x,
            true,
            |x: DType| x * x,
        );
    }

    pub fn _product(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        return self._pool(
            axes,
            keep_dims,
            |x: DType, y: DType| x * y,
            false,
            |x: DType| x,
            false,
            |x: DType| x,
        );
    }

    pub fn _max(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        return self._pool(
            axes,
            keep_dims,
            |x: DType, y: DType| if x > y { x } else { y },
            false,
            |x: DType| x,
            false,
            |x: DType| x,
        );
    }

    pub fn _min(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        return self._pool(
            axes,
            keep_dims,
            |x: DType, y: DType| if x < y { x } else { y },
            false,
            |x: DType| x,
            false,
            |x: DType| x,
        );
    }

    pub fn _reduce_mean(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        let mut pool_size = 1;
        for i in 0..axes.len() {
            pool_size *= self.get_dim_size(axes[i]);
        }

        match DType::from_usize(pool_size) {
            Some(s) => self._pool(
                axes,
                keep_dims,
                |x: DType, y: DType| x + y,
                true,
                |x: DType| x / s,
                false,
                |x: DType| x,
            ),
            None => panic!("Tensor size too large to compute mean for given dtype"),
        }
    }

    pub fn _reduce_mean_square(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        let mut pool_size = 1;
        for i in 0..axes.len() {
            pool_size *= self.get_dim_size(axes[i]);
        }

        match DType::from_usize(pool_size) {
            Some(s) => self._pool(
                axes,
                keep_dims,
                |x: DType, y: DType| (x * x) + y,
                true,
                |x: DType| x / s,
                true,
                |x: DType| x * x,
            ),
            None => panic!("Cant convert from usize to given dtype"),
        }
    }

    pub fn _arg_max(&self, axes: &Vec<usize>, select_last_index: bool) -> Tensor<u32> {
        let result_rank = self.rank() - axes.len() as usize;

        let mut result_shape = vec![0; result_rank + 1];
        let mut result_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        let mut res_ix_map = vec![0; result_rank];
        for i in 0..self.rank() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
            } else {
                result_shape[result_ix] = self.get_dim_size(i);
                res_ix_map[result_ix] = i;
                result_ix += 1;
                result_size *= self.get_dim_size(i);
            }
        }

        let agg_size = result_size;
        result_size *= axes.len();

        result_shape[result_rank] = axes.len();
        let result_strides = compute_strides(&result_shape);
        let mut values: Vec<u32> = vec![zero(); result_size];
        let mut max_vals = vec![zero(); agg_size];
        let mut initialized = vec![false; agg_size];

        let mut input_index = vec![0; self.rank()];
        for i in 0..self.size {
            let mut res_ix = 0;
            for j in 0..result_rank {
                res_ix += result_strides[j] * input_index[res_ix_map[j]];
            }
            let val_ix = res_ix / axes.len();

            if !initialized[val_ix] {
                max_vals[val_ix] = self.get_ix(i);
                initialized[val_ix] = true;
                for j in 0..axes.len() {
                    values[res_ix + j] = input_index[axes[j]] as u32;
                }
            } else {
                let val = self.get_ix(i);
                if max_vals[val_ix] < val || (max_vals[val_ix] == val && select_last_index) {
                    max_vals[val_ix] = val;
                    for j in 0..axes.len() {
                        values[res_ix + j] = input_index[axes[j]] as u32;
                    }
                }
            }

            increment_index(&mut input_index, self.get_sh());
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }

    pub fn _arg_min(&self, axes: &Vec<usize>, select_last_index: bool) -> Tensor<u32> {
        let result_rank = self.rank() - axes.len() as usize;

        let mut result_shape = vec![0; result_rank + 1];
        let mut result_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        let mut res_ix_map = vec![0; result_rank];
        for i in 0..self.rank() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
            } else {
                result_shape[result_ix] = self.get_dim_size(i);
                res_ix_map[result_ix] = i;
                result_ix += 1;
                result_size *= self.get_dim_size(i);
            }
        }

        let agg_size = result_size;
        result_size *= axes.len();

        result_shape[result_rank] = axes.len();
        let result_strides = compute_strides(&result_shape);
        let mut values: Vec<u32> = vec![zero(); result_size];
        let mut max_vals = vec![zero(); agg_size];
        let mut initialized = vec![false; agg_size];

        let mut input_index = vec![0; self.rank()];
        for i in 0..self.size {
            let mut res_ix = 0;
            for j in 0..result_rank {
                res_ix += result_strides[j] * input_index[res_ix_map[j]];
            }
            let val_ix = res_ix / axes.len();

            if !initialized[val_ix] {
                max_vals[val_ix] = self.get_ix(i);
                initialized[val_ix] = true;
                for j in 0..axes.len() {
                    values[res_ix + j] = input_index[axes[j]] as u32;
                }
            } else {
                let val = self.get_ix(i);
                if max_vals[val_ix] > val || (max_vals[val_ix] == val && select_last_index) {
                    max_vals[val_ix] = val;
                    for j in 0..axes.len() {
                        values[res_ix + j] = input_index[axes[j]] as u32;
                    }
                }
            }

            increment_index(&mut input_index, self.get_sh());
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }
}

impl<DType> Tensor<DType>
where
    DType: Clone,
    DType: Num,
    DType: PartialOrd,
    DType: FromPrimitive,
    DType: Float,
{
    pub fn _reduce_log_sum(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        return self._pool(
            axes,
            keep_dims,
            |x: DType, y: DType| x + y,
            true,
            |x: DType| x.ln(),
            false,
            |x: DType| x,
        );
    }

    pub fn _reduce_log_sum_exp(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor<DType> {
        return self._pool(
            axes,
            keep_dims,
            |x: DType, y: DType| x.exp() + y,
            true,
            |x: DType| x.ln(),
            true,
            |x: DType| x.exp(),
        );
    }
}

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: PartialOrd,
    DType: FromPrimitive,
{
    pub fn sum(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._sum(&ax, keep_dims);
    }

    pub fn sum_square(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._sum_square(&ax, keep_dims);
    }

    pub fn product(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._product(&ax, keep_dims);
    }

    pub fn max(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._max(&ax, keep_dims);
    }

    pub fn min(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._min(&ax, keep_dims);
    }

    pub fn reduce_mean(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._reduce_mean(&ax, keep_dims);
    }

    pub fn reduce_mean_square(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._reduce_mean_square(&ax, keep_dims);
    }

    pub fn arg_max(&self, axes: Uint32Array, select_last_index: bool) -> Tensor<u32> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._arg_max(&ax, select_last_index);
    }

    pub fn arg_min(&self, axes: Uint32Array, select_last_index: bool) -> Tensor<u32> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._arg_min(&ax, select_last_index);
    }
}

impl<DType> Tensor<DType>
where
    DType: Clone,
    DType: Num,
    DType: PartialOrd,
    DType: FromPrimitive,
    DType: Float,
{
    pub fn reduce_log_sum(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._reduce_log_sum(&ax, keep_dims);
    }

    pub fn reduce_log_sum_exp(&self, axes: Uint32Array, keep_dims: bool) -> Tensor<DType> {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._reduce_log_sum_exp(&ax, keep_dims);
    }
}

pub fn pool_result(
    in_shape: &Vec<usize>,
    axes: &Vec<usize>,
    keep_dims: bool,
) -> (Vec<usize>, Vec<usize>) {
    let mut result_rank = in_shape.len() - axes.len() as usize;

    if keep_dims {
        result_rank = in_shape.len();
    }
    if result_rank == 0 {
        result_rank = 1
    }

    let mut result_shape = vec![0; result_rank];
    let mut axes_ix = 0;
    let mut result_ix = 0;
    let mut res_ix_map = vec![0; result_rank];
    for i in 0..in_shape.len() {
        if axes_ix < axes.len() && axes[axes_ix] as usize == i {
            axes_ix += 1;
            if keep_dims {
                result_shape[result_ix] = 1;
                res_ix_map[result_ix] = i;
                result_ix += 1;
            }
        } else {
            result_shape[result_ix] = in_shape[i];
            res_ix_map[result_ix] = i;
            result_ix += 1;
        }
    }

    (result_shape, res_ix_map)
}
