use crate::shape::*;
use crate::tensor::*;
use js_sys::Uint32Array;
use num_traits::zero;
use num_traits::Num;

impl Tensor<u32> {
    pub fn _reshape_sparse_indices(
        &self,
        old_sparse_shape: &Vec<usize>,
        new_shape: &Vec<usize>,
    ) -> Tensor<u32> {
        let old_sparse_size = get_size(old_sparse_shape);

        let mut sparse_shape = vec![0; 0];
        let mut sparse_size = 1;
        for i in 0..new_shape.len() {
            if sparse_size < old_sparse_size {
                sparse_size *= new_shape[i];
                sparse_shape.push(new_shape[i]);
            } else {
                break;
            }
        }

        let old_sparse_strides = compute_strides(old_sparse_shape);
        let new_sparse_strides = compute_strides(&sparse_shape);

        let nnz_fraction = sparse_size / old_sparse_size;
        let nnz = self.get_dim_size(0) * nnz_fraction;

        let mut indice_values = vec![0 as u32; nnz * sparse_shape.len()];

        for i in 0..nnz {
            let old_nnz_ix = i / nnz_fraction;

            let mut old_sparse_ix: Vec<usize> = vec![0; old_sparse_shape.len()];
            for j in 0..old_sparse_shape.len() {
                old_sparse_ix[j] = self.get_ix(old_nnz_ix * old_sparse_shape.len() + j) as usize;
            }

            let old_sparse_pos = index_to_pos(&old_sparse_ix, &old_sparse_strides);
            let new_sparse_pos = old_sparse_pos * nnz_fraction + (i % nnz_fraction);
            let new_sparse_ix = pos_to_index(new_sparse_pos, &new_sparse_strides);
            for j in 0..sparse_shape.len() {
                indice_values[i * sparse_shape.len() + j] = new_sparse_ix[j] as u32;
            }
        }

        Tensor::new(
            vec![nnz, sparse_shape.len()],
            vec![sparse_shape.len(), 1],
            nnz * sparse_shape.len(),
            indice_values,
        )
    }

    pub fn _repeat_sparse_indices(
        &self,
        repeats: &Vec<usize>,
        shape: &Vec<usize>,
        repeats_prod: u32,
    ) -> Tensor<u32> {
        let nnz = self.get_dim_size(0);
        let nnz_new = nnz * repeats_prod as usize;
        let s = self.get_dim_size(1);

        let result_shape = vec![nnz_new, s];
        let result_size = nnz_new * s;
        let result_strides = vec![s, 1];
        let mut result_values = vec![0; result_size];

        let mut repeat_ix = vec![0; s];
        for repeat_pos in 0..repeats_prod {
            for i in 0..nnz {
                for j in 0..s {
                    result_values[(repeat_pos as usize * nnz + i) * s + j] =
                        repeat_ix[j] as u32 * shape[j] as u32 + self.get_ix(i * s + j);
                }
            }

            increment_index(&mut repeat_ix, repeats);
        }

        Tensor::new(result_shape, result_strides, result_size, result_values)
    }
}

impl Tensor<u32> {
    pub fn reshape_sparse_indices(
        &self,
        old_sparse_shape: Uint32Array,
        new_shape: Uint32Array,
    ) -> Tensor<u32> {
        let mut _new_shape: Vec<usize> = vec![0; new_shape.length() as usize];
        for i in 0..new_shape.length() {
            _new_shape[i as usize] = new_shape.get_index(i) as usize;
        }

        let mut _old_sparse_shape: Vec<usize> = vec![0; old_sparse_shape.length() as usize];
        for i in 0..old_sparse_shape.length() {
            _old_sparse_shape[i as usize] = old_sparse_shape.get_index(i) as usize;
        }

        self._reshape_sparse_indices(&_old_sparse_shape, &_new_shape)
    }

    pub fn add_index(&self, axis: i32, count: i32) -> Tensor<u32> {
        let result_shape = self.get_sh().to_vec();
        let result_strides = self.get_strides().to_vec();
        let result_size = self.size;
        let mut result_values = self.get_values().to_vec();

        for i in ((axis as usize)..result_size).step_by(result_shape[1]) {
            result_values[i] += count as u32;
        }

        Tensor::new(result_shape, result_strides, result_size, result_values)
    }

    pub fn repeat_sparse_indices(
        &self,
        repeats: Uint32Array,
        shape: Uint32Array,
        repeats_prod: u32,
    ) -> Tensor<u32> {
        let mut _repeats: Vec<usize> = vec![0; repeats.length() as usize];
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..repeats.length() {
            _repeats[i as usize] = repeats.get_index(i) as usize;
            _shape[i as usize] = shape.get_index(i) as usize;
        }

        self._repeat_sparse_indices(&_repeats, &_shape, repeats_prod)
    }
}

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
}
