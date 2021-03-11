use crate::shape::*;
use crate::tensor::*;
use js_sys::Uint32Array;

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
}
