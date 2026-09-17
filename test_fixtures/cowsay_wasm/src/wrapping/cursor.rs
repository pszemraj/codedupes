use super::{fragments::segment, shared::display_width};

/// Wrap paragraphs through indexed token and character cursors.
pub(crate) fn wrap(message: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut output = Vec::new();
    for raw_paragraph in message.split('\n') {
        let paragraph = raw_paragraph.strip_suffix('\r').unwrap_or(raw_paragraph);
        let words: Vec<&str> = paragraph.split_whitespace().collect();
        if words.is_empty() {
            output.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut word_index = 0;
        while word_index < words.len() {
            let fragments = segment(words[word_index], width);
            let mut fragment_index = 0;
            while fragment_index < fragments.len() {
                let fragment = &fragments[fragment_index];
                let separator = usize::from(!line.is_empty());
                if display_width(&line) + separator + display_width(fragment) <= width {
                    if separator == 1 {
                        line.push(' ');
                    }
                    line.push_str(fragment);
                } else {
                    output.push(std::mem::take(&mut line));
                    line.push_str(fragment);
                }
                fragment_index += 1;
            }
            word_index += 1;
        }
        if !line.is_empty() {
            output.push(line);
        }
    }
    if output.is_empty() {
        output.push(String::new());
    }
    output
}
