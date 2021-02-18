use crate::utils::conv_output_size;
use js_sys::Int32Array;

use wasm_bindgen::prelude::*;

use js_sys::{Float32Array, Uint32Array};

use crate::shape::*;
use crate::tensor::*;
use crate::utils::*;

impl Tensor {
    pub fn _reshape(&self, shape: &Vec<usize>) -> Tensor {
        let strides = compute_strides(shape);

        let mut values = vec![0.0; self.size];
        for i in 0..self.size {
            values[i] = self.get_ix(i);
        }

        Tensor::new(shape.to_vec(), strides, self.size, values)
    }

    pub fn _transpose(&self, permutation: &Vec<usize>) -> Tensor {
        let rank = self.rank();

        let mut output_shape = vec![0; rank];
        let mut reverse_perm = vec![0; rank];
        for i in 0..rank {
            output_shape[i] = self.get_dim_size(permutation[i]);
            reverse_perm[permutation[i]] = i;
        }

        let output_strides = compute_strides(&output_shape);
        let mut mapped_strides = vec![0; rank];
        for i in 0..rank {
            mapped_strides[i] = output_strides[reverse_perm[i]];
        }

        let mut values = vec![0.0; self.size];

        let mut index = vec![0; rank];
        for i in 0..self.size {
            let mut out_ix = 0;
            for j in 0..rank {
                out_ix += index[j] * mapped_strides[j];
            }

            values[out_ix] = self.get_ix(i);

            increment_index(&mut index, self.get_sh());
        }

        Tensor::new(output_shape, output_strides, self.size, values)
    }

    pub fn _repeat(&self, repeats: &Vec<usize>) -> Tensor {
        let rank = self.rank();

        let mut output_shape = vec![0; rank];
        for i in 0..rank {
            output_shape[i] = self.get_dim_size(i) * repeats[i];
        }

        let output_strides = compute_strides(&output_shape);
        let output_size = get_size(&output_shape);

        let mut values = vec![0.0; output_size];

        let mut index = vec![0; rank];
        let mut in_ix = vec![0; rank];
        for i in 0..output_size {
            for j in 0..rank {
                in_ix[j] = index[j] % self.get_dim_size(j);
            }

            values[i] = self.get(&in_ix);

            increment_index(&mut index, &output_shape);
        }

        Tensor::new(output_shape, output_strides, output_size, values)
    }

    pub fn _expand(&self, shape: &Vec<usize>) -> Tensor {
        let mut output_shape = vec![0; shape.len()];
        for i in 0..shape.len() {
            output_shape[i] = shape[i]
        }
        let output_strides = compute_strides(&output_shape);
        let output_size = get_size(&output_shape);

        let mut values = vec![0.0; output_size];

        let mut ix = vec![0; shape.len()];
        for i in 0..output_size {
            values[i] = self.get(&ix);

            increment_index(&mut ix, &output_shape);
        }

        Tensor::new(output_shape, output_strides, output_size, values)
    }

    pub fn _slice(&self, starts: &Vec<usize>, ends: &Vec<usize>, axis: &Vec<usize>) -> Tensor {
        let rank = self.rank();
        let mut result_shape = vec![0; rank];
        let mut ax_ix = 0;
        for i in 0..rank {
            if ax_ix < axis.len() && i == axis[ax_ix] {
                result_shape[i] = ends[ax_ix] - starts[ax_ix];
                ax_ix += 1;
            } else {
                result_shape[i] = self.get_dim_size(i);
            }
        }

        let result_strides = compute_strides(&result_shape);
        let result_size = get_size(&result_shape);

        let mut values = vec![0.0; result_size];

        let mut out_ix = vec![0; rank];
        let mut in_ix = vec![0; rank];

        for i in 0..result_size {
            ax_ix = 0;
            for j in 0..rank {
                if ax_ix < axis.len() && j == axis[ax_ix] {
                    in_ix[j] = out_ix[j] + starts[ax_ix];
                    ax_ix += 1;
                } else {
                    in_ix[j] = out_ix[j];
                }
            }

            values[i] = self.get(&in_ix);

            increment_index(&mut out_ix, &result_shape);
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }

    pub fn _set_values(&self, value_tensor: &Tensor, starts: &Vec<usize>) -> Tensor {
        let rank = self.rank();
        let mut result_shape = vec![0; rank];
        for i in 0..rank {
            result_shape[i] = self.get_dim_size(i);
        }

        let result_strides = compute_strides(&result_shape);
        let result_size = get_size(&result_shape);

        let mut values = vec![0.0; result_size];

        let mut out_ix = vec![0; rank];
        let mut values_ix = vec![0; rank];

        for i in 0..result_size {
            let mut in_values = true;
            for j in 0..rank {
                if out_ix[j] < starts[j] || out_ix[j] >= (value_tensor.get_dim_size(j) + starts[j])
                {
                    in_values = false;
                    break;
                } else {
                    values_ix[j] = out_ix[j] - starts[j];
                }
            }

            if in_values {
                values[i] = value_tensor.get(&values_ix);
            } else {
                values[i] = self.get_ix(i);
            }

            increment_index(&mut out_ix, &result_shape);
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }
}

#[wasm_bindgen]
impl Tensor {
    pub fn set_values(&self, values: &Tensor, starts: Uint32Array) -> Tensor {
        let mut _starts: Vec<usize> = vec![0; starts.length() as usize];
        for i in 0..starts.length() {
            _starts[i as usize] = starts.get_index(i) as usize;
        }
        return self._set_values(values, &_starts);
    }

    pub fn reshape(&self, shape: Uint32Array) -> Tensor {
        let mut sh: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..shape.length() {
            sh[i as usize] = shape.get_index(i) as usize;
        }
        return self._reshape(&sh);
    }

    pub fn concat(&self, other: &Tensor, axes: u32) -> Tensor {
        let ax = axes as usize;
        let mut output_shape = vec![0; self.rank()];
        for i in 0..self.rank() {
            output_shape[i] = self.get_dim_size(i);
        }
        output_shape[ax] += other.get_dim_size(ax);

        let output_strides = compute_strides(&output_shape);
        let output_size = get_size(&output_shape);

        let mut values = vec![0.0; output_size];

        let mut index_x = 0;
        let mut index_y = 0;
        let mut index_out = 0;
        let iter_x_size = output_strides[ax] * self.get_dim_size(ax);
        let iter_y_size = output_strides[ax] * other.get_dim_size(ax);
        let outer_iters = output_size
            / (if ax > 0 {
                output_strides[ax - 1]
            } else {
                output_size
            });
        for _ in 0..outer_iters {
            for _ in 0..iter_x_size {
                values[index_out] = self.get_ix(index_x);
                index_x += 1;
                index_out += 1;
            }

            for _ in 0..iter_y_size {
                values[index_out] = other.get_ix(index_y);
                index_y += 1;
                index_out += 1;
            }
        }

        Tensor::new(output_shape, output_strides, output_size, values)
    }

    pub fn transpose(&self, permutation: Uint32Array) -> Tensor {
        let mut perm: Vec<usize> = vec![0; permutation.length() as usize];
        for i in 0..permutation.length() {
            perm[i as usize] = permutation.get_index(i) as usize;
        }
        return self._transpose(&perm);
    }

    pub fn repeat(&self, repeats: Uint32Array) -> Tensor {
        let mut _repeats: Vec<usize> = vec![0; repeats.length() as usize];
        for i in 0..repeats.length() {
            _repeats[i as usize] = repeats.get_index(i) as usize;
        }
        return self._repeat(&_repeats);
    }

    pub fn expand(&self, shape: Uint32Array) -> Tensor {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..shape.length() {
            _shape[i as usize] = shape.get_index(i) as usize;
        }
        return self._expand(&_shape);
    }

    pub fn copy(&self) -> Tensor {
        let mut _shape: Vec<usize> = vec![0; self.rank() as usize];
        let mut _strides: Vec<usize> = vec![0; self.rank() as usize];
        for i in 0..self.rank() {
            _shape[i] = self.get_dim_size(i);
            _strides[i] = self.get_strides_at(i);
        }

        let mut values = vec![0.0; self.size];
        for i in 0..self.size {
            values[i] = self.get_ix(i);
        }

        Tensor::new(_shape, _strides, self.size, values)
    }

    // Mode: 0 == constant, 1 == reflect, 2 == edge
    pub fn gather(&self, axis: i32, indices: Int32Array, indice_shape: Uint32Array) -> Tensor {
        let indice_strides = compute_strides_uint32(&indice_shape);

        let r = self.rank();
        let q = indice_shape.length();

        let result_rank = r + (q as usize) - 1;
        let mut result_shape = vec![0; result_rank];
        for i in 0..axis {
            result_shape[i as usize] = self.get_dim_size(i as usize);
        }
        for i in 0..q {
            result_shape[(i + (axis as u32)) as usize] = indice_shape.get_index(i) as usize;
        }
        for i in (axis + 1)..(r as i32) {
            result_shape[(i as usize) + (q as usize) - 1] = self.get_dim_size(i as usize);
        }

        let result_strides = compute_strides(&result_shape);
        let result_size = get_size(&result_shape);

        let mut values = vec![0.0; result_size];

        let mut out_ix = vec![0; result_rank];
        let mut input_ix = vec![0; self.rank()];
        for i in 0..result_size {
            let mut gather_pos = 0;
            for j in 0..q {
                gather_pos += out_ix[j as usize + (axis as usize)] * indice_strides[j as usize];
            }
            let ax_ix = indices.get_index(gather_pos as u32);

            for j in 0..axis {
                input_ix[j as usize] = out_ix[j as usize];
            }
            input_ix[axis as usize] = ax_ix as usize;
            for j in (axis as usize + 1)..r {
                input_ix[j] = out_ix[j + q as usize - 1];
            }

            values[i] = self.get(&input_ix);

            increment_index(&mut out_ix, &result_shape);
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }

    pub fn slice(&self, starts: Uint32Array, ends: Uint32Array, axis: Uint32Array) -> Tensor {
        let mut _starts: Vec<usize> = vec![0; starts.length() as usize];
        let mut _ends: Vec<usize> = vec![0; ends.length() as usize];
        let mut _axis: Vec<usize> = vec![0; axis.length() as usize];
        for i in 0..starts.length() {
            _starts[i as usize] = starts.get_index(i) as usize;
            _ends[i as usize] = ends.get_index(i) as usize;
            _axis[i as usize] = axis.get_index(i) as usize;
        }

        return self._slice(&_starts, &_ends, &_axis);
    }
}
