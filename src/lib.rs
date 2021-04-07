#![cfg_attr(feature = "benchmark", feature(test))]
pub mod shape;
pub mod tensor;
pub mod utils;

pub mod dtype {
    pub mod f32;
    pub mod f64;
    pub mod i16;
    pub mod i32;
    pub mod i8;
    pub mod u16;
    pub mod u32;
    pub mod u8;
}

pub mod ops {
    pub mod aggregate;
    pub mod backward;
    pub mod binary;
    pub mod conv;
    pub mod matmul;
    pub mod unary;
    pub mod util;
    pub mod sparse {
        pub mod aggregate;
        pub mod binary;
        pub mod matmul;
        pub mod util;
    }
}

pub mod tests {
    pub mod shape;
    pub mod tensor {
        pub mod aggregate;
        pub mod basic;

        #[cfg(feature = "benchmark")]
        pub mod benchmark;
        pub mod conv;
        pub mod pool;
        pub mod sparse;
        pub mod tensor;
    }
}

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
