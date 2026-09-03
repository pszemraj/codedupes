mod speech;
mod thought;

use crate::wrapping::display_width;

pub(crate) fn render(lines: &[String], thinking: bool) -> String {
    let bubble_width = lines
        .iter()
        .map(|line| display_width(line))
        .max()
        .unwrap_or(0);

    if thinking {
        thought::render_bubble(lines, bubble_width)
    } else {
        speech::render_bubble(lines, bubble_width)
    }
}

pub(crate) fn pad_line(line: &str, width: usize) -> String {
    let padding = width.saturating_sub(display_width(line));
    let mut padded = String::with_capacity(line.len() + padding);
    padded.push_str(line);
    padded.extend(std::iter::repeat_n(' ', padding));
    padded
}
