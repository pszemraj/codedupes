use crate::{bubble, cow, wrapping, CowOptions};

/// Assemble a completed cowsay response through a mutable output buffer.
///
/// This alternate rendering path deliberately owns its layout preparation so
/// callers can select it independently from the format-based pipeline.
pub(crate) fn render(message: &str, options: CowOptions) -> String {
    let columns = options
        .width
        .clamp(wrapping::MIN_WRAP_WIDTH, wrapping::MAX_WRAP_WIDTH);
    let rows = wrapping::wrap(message, columns, options.wrap_algorithm);
    let mut response = bubble::render(&rows, options.thinking);
    response.push('\n');
    response.push_str(&cow::render(options.thinking));
    response
}
