use crate::shape::*;
use crate::tensor::*;
use wasm_bindgen::prelude::*;

use std::cmp;

impl Tensor {
    pub fn _gemm(
        &self,
        b: &Tensor,
        a_transpose: bool,
        b_transpose: bool,
        alpha: f32,
        c: Option<&Tensor>,
        beta: f32,
    ) -> Tensor {
        let rank = self.rank();

        let M = if a_transpose {
            self.get_dim_size(rank - 1)
        } else {
            self.get_dim_size(rank - 2)
        };
        let N = if a_transpose {
            self.get_dim_size(rank - 2)
        } else {
            self.get_dim_size(rank - 1)
        };
        let O = if b_transpose {
            b.get_dim_size(rank - 2)
        } else {
            b.get_dim_size(rank - 1)
        };

        let a_batch_mult = M * N;
        let b_batch_mult = N * O;
        let y_batch_mult = M * O;

        let a_n_mult = if a_transpose { M } else { 1 };
        let a_m_mult = if a_transpose { 1 } else { N };
        let b_n_mult = if b_transpose { 1 } else { O };
        let b_o_mult = if b_transpose { N } else { 1 };

        let mut c_m_mult = 0;
        let mut c_o_mult = 0;
        let mut c_batch_mult = 0;
        match c {
            None => {}
            Some(c_) => {
                let c_batch_size = get_size_from_to(c_.get_sh(), 0, rank - 2);

                c_m_mult = c_.get_strides_at(rank - 2);
                c_o_mult = c_.get_strides_at(rank - 1);
                if c_batch_size > 1 {
                    c_batch_mult = c_.get_dim_size(rank - 2) * c_.get_dim_size(rank - 1);

                    if c_batch_mult == 1 {
                        c_batch_mult = 0;
                    }
                } else {
                    c_batch_mult = 0;
                }
            }
        }

        let mut batch_size = get_size_from_to(self.get_sh(), 0, rank - 2);
        if batch_size == 0 {
            batch_size = 1;
        }
        let mut result_shape = vec![0; rank];
        result_shape[rank - 1] = O;
        result_shape[rank - 2] = M;
        for i in 0..(rank - 2) {
            result_shape[i] = self.get_dim_size(i);
        }
        let result_size = get_size(&result_shape);
        let result_strides = compute_strides(&result_shape);

        let mut values = vec![0.0; result_size];

        for i in 0..batch_size {
            let a_base = i * a_batch_mult;
            let b_base = i * b_batch_mult;
            let y_base = i * y_batch_mult;
            let c_base = i * c_batch_mult;

            for m in 0..M {
                for o in 0..O {
                    let mut result = 0.0;

                    for n in 0..N {
                        result += self.get_ix(a_base + m * a_m_mult + n * a_n_mult)
                            * b.get_ix(b_base + n * b_n_mult + o * b_o_mult);
                    }

                    result = alpha * result;
                    match c {
                        None => {}
                        Some(c_) => {
                            let ix = c_base + m * c_m_mult + o * c_o_mult;
                            result += beta * c_.get_ix(ix);
                        }
                    }
                    values[y_base + m * O + o] = result;
                }
            }
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }
}

#[wasm_bindgen]
impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let m = self.get_dim_size(0);
        let n = self.get_dim_size(1);
        let o = other.get_dim_size(1);

        let mut values = vec![0.; m * o];
        for i in 0..m {
            for k in 0..o {
                let mut res = 0.;
                for j in 0..n {
                    res += self.get_ix(i * n + j) * other.get_ix(j * o + k);
                }
                values[i * o + k] = res;
            }
        }

        Tensor::new(vec![m, o], vec![o, 1], m * o, values)
    }

    pub fn gemm(&self, other: &Tensor, a_transpose: bool, b_transpose: bool, alpha: f32) -> Tensor {
        return self._gemm(other, a_transpose, b_transpose, alpha, None, 1.0);
    }

    pub fn gemm_with_c(
        &self,
        other: &Tensor,
        a_transpose: bool,
        b_transpose: bool,
        alpha: f32,
        c: &Tensor,
        beta: f32,
    ) -> Tensor {
        return self._gemm(other, a_transpose, b_transpose, alpha, Some(c), beta);
    }
}
