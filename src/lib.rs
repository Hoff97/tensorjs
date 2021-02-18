pub mod shape;
pub mod tensor;
pub mod utils;

pub mod ops {
    pub mod aggregate;
    pub mod binary;
    pub mod conv;
    pub mod matmul;
    pub mod unary;
    pub mod util;
}

pub mod tests {
    pub mod shape;
    pub mod tensor {
        pub mod aggregate;
        pub mod basic;
        pub mod conv;
        pub mod pool;
        pub mod tensor;
    }
}

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
