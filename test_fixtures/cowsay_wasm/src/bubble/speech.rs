use super::pad_line;

// fixture:exact-border:start
pub(crate) fn make_borders(width: usize) -> (String, String) {
    let horizontal_span = width + 2;
    let capacity = horizontal_span + 1;

    let mut top = String::with_capacity(capacity);
    top.push(' ');
    top.push_str(&"_".repeat(horizontal_span));

    let mut bottom = String::with_capacity(capacity);
    bottom.push(' ');
    bottom.push_str(&"-".repeat(horizontal_span));

    (top, bottom)
}
// fixture:exact-border:end

// fixture:edit-bubble:start
pub(crate) fn render_bubble(lines: &[String], width: usize) -> String {
    let (top_border, bottom_border) = make_borders(width);
    let mut output = String::new();
    output.push_str(&top_border);
    output.push('\n');

    for (index, line) in lines.iter().enumerate() {
        let padded = pad_line(line, width);
        let (left, right) = match (lines.len(), index) {
            (1, _) => ('<', '>'),
            (_, 0) => ('/', '\\'),
            (line_count, current) if current + 1 == line_count => ('\\', '/'),
            _ => ('|', '|'),
        };

        output.push(left);
        output.push(' ');
        output.push_str(&padded);
        output.push(' ');
        output.push(right);
        output.push('\n');
    }

    output.push_str(&bottom_border);
    output
}
// fixture:edit-bubble:end
