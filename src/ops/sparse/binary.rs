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
    pub fn binary_sparse_dense<F>(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: &Vec<usize>,
        op: F,
    ) -> Tensor<DType>
    where
        F: Fn(DType, DType) -> DType,
    {
        let s = indices.get_dim_size(1);
        let nnz = indices.get_dim_size(0);

        let d = result_shape.len() - s;

        let mut result_values_shape = vec![0; d + 1];
        result_values_shape[0] = nnz;
        for i in 0..d {
            result_values_shape[i + 1] = result_shape[s + i];
        }
        let result_values_strides = compute_strides(&result_values_shape);
        let result_values_size = get_size(&result_values_shape);
        let mut result_values = vec![zero(); result_values_size];

        let dense_size = result_values_strides[0];

        for i in 0..nnz {
            let mut result_ix: Vec<usize> = vec![0; result_shape.len()];
            for j in 0..s {
                result_ix[j] = indices.get_ix(i * s + j) as usize;
            }

            for j in 0..dense_size {
                let mut ix_a = vec![0; d + 1];
                ix_a[0] = i;
                for k in 0..d {
                    ix_a[k + 1] = result_ix[s + k];
                }
                let v_a = self.get(&ix_a);
                let v_b = b.get(&result_ix);

                result_values[i * dense_size + j] = op(v_a, v_b);

                increment_index(&mut result_ix, &result_shape);
            }
        }

        Tensor::new(
            result_values_shape,
            result_values_strides,
            result_values_size,
            result_values,
        )
    }

    pub fn binary_sparse_sparse<F>(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: &Vec<usize>,
        op: F,
    ) -> Tensor<DType>
    where
        F: Fn(DType, DType) -> DType,
    {
        let s = indices.get_dim_size(1);
        let nnz = indices.get_dim_size(0);

        let d = result_shape.len() - s;

        let mut result_values_shape = vec![0; d + 1];
        result_values_shape[0] = nnz;
        for i in 0..d {
            result_values_shape[i + 1] = result_shape[s + i];
        }
        let result_values_strides = compute_strides(&result_values_shape);
        let result_values_size = get_size(&result_values_shape);
        let mut result_values = vec![zero(); result_values_size];

        let mut sparse_shape = vec![0; s];
        for i in 0..s {
            sparse_shape[i] = result_shape[i];
        }
        let sparse_strides = compute_strides(&sparse_shape);

        let dense_size = result_values_strides[0];

        let mut b_position_map: HashMap<usize, usize> = HashMap::with_capacity(nnz);
        for i in 0..nnz {
            let mut pos = 0;
            for j in 0..s {
                pos += b_indices.get_ix(i * s + j) as usize * sparse_strides[j];
            }
            b_position_map.insert(pos, i);
        }

        for i in 0..nnz {
            let mut pos = 0;
            for j in 0..s {
                pos += indices.get_ix(i * s + j) as usize * sparse_strides[j];
            }
            let i_b = b_position_map[&pos];

            let mut dense_ix = vec![0; d + 1];

            for j in 0..dense_size {
                dense_ix[0] = i;
                let v_a = self.get(&dense_ix);
                dense_ix[0] = i_b;
                let v_b = b_values.get(&dense_ix);

                result_values[i * dense_size + j] = op(v_a, v_b);

                increment_index(&mut dense_ix, &result_shape);
            }
        }

        Tensor::new(
            result_values_shape,
            result_values_strides,
            result_values_size,
            result_values,
        )
    }

    pub fn _add_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: &Vec<usize>,
        alpha: DType,
        beta: DType,
    ) -> Tensor<DType> {
        self.binary_sparse_dense(indices, b, result_shape, |a: DType, b: DType| {
            alpha * a + beta * b
        })
    }

    pub fn _subtract_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: &Vec<usize>,
        alpha: DType,
        beta: DType,
    ) -> Tensor<DType> {
        self.binary_sparse_dense(indices, b, result_shape, |a: DType, b: DType| {
            alpha * a - beta * b
        })
    }

    pub fn _multiply_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: &Vec<usize>,
        alpha: DType,
    ) -> Tensor<DType> {
        self.binary_sparse_dense(indices, b, result_shape, |a: DType, b: DType| alpha * a * b)
    }

    pub fn _divide_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: &Vec<usize>,
        alpha: DType,
    ) -> Tensor<DType> {
        self.binary_sparse_dense(indices, b, result_shape, |a: DType, b: DType| alpha * a / b)
    }

    pub fn add_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: Uint32Array,
        alpha: DType,
        beta: DType,
    ) -> Tensor<DType> {
        let l = result_shape.length() as usize;
        let mut _result_shape = vec![0; l];
        for i in 0..l {
            _result_shape[i] = result_shape.get_index(i as u32) as usize;
        }

        self._add_sparse_dense(indices, b, &_result_shape, alpha, beta)
    }

    pub fn subtract_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: Uint32Array,
        alpha: DType,
        beta: DType,
    ) -> Tensor<DType> {
        let l = result_shape.length() as usize;
        let mut _result_shape = vec![0; l];
        for i in 0..l {
            _result_shape[i] = result_shape.get_index(i as u32) as usize;
        }

        self._subtract_sparse_dense(indices, b, &_result_shape, alpha, beta)
    }

    pub fn multiply_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: Uint32Array,
        alpha: DType,
    ) -> Tensor<DType> {
        let l = result_shape.length() as usize;
        let mut _result_shape = vec![0; l];
        for i in 0..l {
            _result_shape[i] = result_shape.get_index(i as u32) as usize;
        }

        self._multiply_sparse_dense(indices, b, &_result_shape, alpha)
    }

    pub fn divide_sparse_dense(
        &self,
        indices: &Tensor<u32>,
        b: &Tensor<DType>,
        result_shape: Uint32Array,
        alpha: DType,
    ) -> Tensor<DType> {
        let l = result_shape.length() as usize;
        let mut _result_shape = vec![0; l];
        for i in 0..l {
            _result_shape[i] = result_shape.get_index(i as u32) as usize;
        }

        self._divide_sparse_dense(indices, b, &_result_shape, alpha)
    }

    pub fn _add_sparse_sparse(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: &Vec<usize>,
        alpha: DType,
        beta: DType,
    ) -> Tensor<DType> {
        self.binary_sparse_sparse(
            indices,
            b_indices,
            b_values,
            result_shape,
            |a: DType, b: DType| alpha * a + beta * b,
        )
    }

    pub fn add_sparse_sparse(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: Uint32Array,
        alpha: DType,
        beta: DType,
    ) -> Tensor<DType> {
        let l = result_shape.length() as usize;
        let mut _result_shape = vec![0; l];
        for i in 0..l {
            _result_shape[i] = result_shape.get_index(i as u32) as usize;
        }

        self._add_sparse_sparse(indices, b_indices, b_values, &_result_shape, alpha, beta)
    }

    pub fn _subtract_sparse_sparse(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: &Vec<usize>,
        alpha: DType,
        beta: DType,
    ) -> Tensor<DType> {
        self.binary_sparse_sparse(
            indices,
            b_indices,
            b_values,
            result_shape,
            |a: DType, b: DType| alpha * a - beta * b,
        )
    }

    pub fn subtract_sparse_sparse(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: Uint32Array,
        alpha: DType,
        beta: DType,
    ) -> Tensor<DType> {
        let l = result_shape.length() as usize;
        let mut _result_shape = vec![0; l];
        for i in 0..l {
            _result_shape[i] = result_shape.get_index(i as u32) as usize;
        }

        self._subtract_sparse_sparse(indices, b_indices, b_values, &_result_shape, alpha, beta)
    }

    pub fn _multiply_sparse_sparse(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: &Vec<usize>,
        alpha: DType,
    ) -> Tensor<DType> {
        self.binary_sparse_sparse(
            indices,
            b_indices,
            b_values,
            result_shape,
            |a: DType, b: DType| alpha * a * b,
        )
    }

    pub fn multiply_sparse_sparse(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: Uint32Array,
        alpha: DType,
    ) -> Tensor<DType> {
        let l = result_shape.length() as usize;
        let mut _result_shape = vec![0; l];
        for i in 0..l {
            _result_shape[i] = result_shape.get_index(i as u32) as usize;
        }

        self._multiply_sparse_sparse(indices, b_indices, b_values, &_result_shape, alpha)
    }

    pub fn _divide_sparse_sparse(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: &Vec<usize>,
        alpha: DType,
    ) -> Tensor<DType> {
        self.binary_sparse_sparse(
            indices,
            b_indices,
            b_values,
            result_shape,
            |a: DType, b: DType| alpha * a / b,
        )
    }

    pub fn divide_sparse_sparse(
        &self,
        indices: &Tensor<u32>,
        b_indices: &Tensor<u32>,
        b_values: &Tensor<DType>,
        result_shape: Uint32Array,
        alpha: DType,
    ) -> Tensor<DType> {
        let l = result_shape.length() as usize;
        let mut _result_shape = vec![0; l];
        for i in 0..l {
            _result_shape[i] = result_shape.get_index(i as u32) as usize;
        }

        self._divide_sparse_sparse(indices, b_indices, b_values, &_result_shape, alpha)
    }
}
