use super::shared::{display_width, split_word};

// fixture:semantic-wrap:start
pub(crate) fn wrap(message: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut wrapped = Vec::new();

    for raw_paragraph in message.split('\n') {
        let paragraph = raw_paragraph.strip_suffix('\r').unwrap_or(raw_paragraph);
        if paragraph.split_whitespace().next().is_none() {
            wrapped.push(String::new());
            continue;
        }

        let mut active = String::new();
        let mut active_width = 0;

        for word in paragraph.split_whitespace() {
            for piece in split_word(word, width) {
                let piece_width = display_width(&piece);
                let separator_width = if active.is_empty() { 0 } else { 1 };

                if active_width + separator_width + piece_width <= width {
                    if separator_width == 1 {
                        active.push(' ');
                        active_width += 1;
                    }
                    active.push_str(&piece);
                    active_width += piece_width;
                } else {
                    wrapped.push(std::mem::take(&mut active));
                    active.push_str(&piece);
                    active_width = piece_width;
                }
            }
        }

        if !active.is_empty() {
            wrapped.push(active);
        }
    }

    if wrapped.is_empty() {
        wrapped.push(String::new());
    }

    wrapped
}
// fixture:semantic-wrap:end
