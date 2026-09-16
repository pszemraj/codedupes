//! Cowsay implemented in Rust, with a small WebAssembly boundary.
//!
//! The scanner and fold algorithms are both supported rendering paths.

mod bubble;
mod cow;
mod wrapping;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub use wrapping::WrapAlgorithm;

/// Rendering options for the native Rust API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CowOptions {
    /// Maximum message width before word wrapping. Values are clamped to 4..=96.
    pub width: usize,
    /// Render a thought bubble and thought connector instead of speech.
    pub thinking: bool,
    /// Select the wrapping implementation.
    pub wrap_algorithm: WrapAlgorithm,
}

impl Default for CowOptions {
    fn default() -> Self {
        Self {
            width: 40,
            thinking: false,
            wrap_algorithm: WrapAlgorithm::Scanner,
        }
    }
}

/// Render a complete cowsay string through the native Rust API.
pub fn render(message: &str, options: CowOptions) -> String {
    let width = options
        .width
        .clamp(wrapping::MIN_WRAP_WIDTH, wrapping::MAX_WRAP_WIDTH);
    let lines = wrapping::wrap(message, width, options.wrap_algorithm);
    let bubble = bubble::render(&lines, options.thinking);
    let cow = cow::render(options.thinking);

    format!("{bubble}\n{cow}")
}

/// Browser-facing WebAssembly API.
///
/// `use_fold_wrapper` selects the fold implementation instead of the scanner.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn cowsay(message: &str, width: u32, thinking: bool, use_fold_wrapper: bool) -> String {
    let wrap_algorithm = if use_fold_wrapper {
        WrapAlgorithm::Fold
    } else {
        WrapAlgorithm::Scanner
    };

    render(
        message,
        CowOptions {
            width: width as usize,
            thinking,
            wrap_algorithm,
        },
    )
}

/// Exposed so the browser demo can show which Rust package it loaded.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn fixture_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}
