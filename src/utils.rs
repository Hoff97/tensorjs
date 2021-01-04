use js_sys::{Uint32Array};

pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[macro_export]
macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if !($x - $y < $d || $y - $x < $d) { panic!(); }
    }
}

pub fn uint32_array(vec: &Vec<usize>) -> Uint32Array {
    let result = Uint32Array::new_with_length(vec.len() as u32);
    for i in 0..vec.len() {
        result.set_index(i as u32, vec[i] as u32);
    }
    result
}

pub fn conv_output_size(in_sizes: &Vec<usize>,
                        kernels: &Vec<usize>,
                        pads: &Vec<usize>,
                        dilations: &Vec<usize>,
                        strides: &Vec<usize>) -> Vec<usize> {
    let mut result = vec![0; dilations.len()];
    for i in 0..dilations.len() {
        let dkernel = dilations[i] * (kernels[i+2] - 1) + 1;
        result[i] =  ((in_sizes[i+2] + pads[i] + pads[i + dilations.len()] - dkernel) / strides[i]) + 1
    }
    result
}
