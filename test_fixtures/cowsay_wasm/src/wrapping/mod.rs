mod fold;
mod scanner;
mod shared;

pub(crate) use shared::display_width;

pub(crate) const MIN_WRAP_WIDTH: usize = 4;
pub(crate) const MAX_WRAP_WIDTH: usize = 96;

/// The two variants are intentionally semantically equivalent but implemented
/// with different control flow for the clone fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapAlgorithm {
    Scanner,
    Fold,
}

pub(crate) fn wrap(message: &str, width: usize, algorithm: WrapAlgorithm) -> Vec<String> {
    match algorithm {
        WrapAlgorithm::Scanner => scanner::wrap(message, width),
        WrapAlgorithm::Fold => fold::wrap(message, width),
    }
}
