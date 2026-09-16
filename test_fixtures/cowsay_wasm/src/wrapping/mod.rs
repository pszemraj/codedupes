mod fold;
mod fragments;
mod queue;
mod scanner;
mod shared;

pub(crate) use shared::display_width;

pub(crate) const MIN_WRAP_WIDTH: usize = 4;
pub(crate) const MAX_WRAP_WIDTH: usize = 96;

/// Select a native text wrapping implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapAlgorithm {
    Scanner,
    Fold,
    Queue,
}

pub(crate) fn wrap(message: &str, width: usize, algorithm: WrapAlgorithm) -> Vec<String> {
    match algorithm {
        WrapAlgorithm::Scanner => scanner::wrap(message, width),
        WrapAlgorithm::Fold => fold::wrap(message, width),
        WrapAlgorithm::Queue => queue::wrap(message, width),
    }
}
