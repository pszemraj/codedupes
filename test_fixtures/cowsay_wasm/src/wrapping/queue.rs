use std::collections::VecDeque;

use super::{fragments::segment, shared::display_width};

/// Wrap words by consuming a queue of pending text.
pub(crate) fn wrap(message: &str, width: usize) -> Vec<String> {
    let limit = width.max(1);
    let mut output = Vec::new();

    for raw_paragraph in message.split('\n') {
        let paragraph = raw_paragraph.strip_suffix('\r').unwrap_or(raw_paragraph);
        let mut pending: VecDeque<String> =
            paragraph.split_whitespace().map(str::to_owned).collect();

        if pending.is_empty() {
            output.push(String::new());
            continue;
        }

        let mut line = String::new();
        while let Some(word) = pending.pop_front() {
            if display_width(&word) > limit {
                if !line.is_empty() {
                    output.push(std::mem::take(&mut line));
                }
                let mut pieces = segment(&word, limit);
                let tail = pieces.pop().expect("long word has a final piece");
                output.extend(pieces);
                line = tail;
                continue;
            }

            let additional = if line.is_empty() {
                display_width(&word)
            } else {
                1 + display_width(&word)
            };
            if !line.is_empty() && display_width(&line) + additional > limit {
                output.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&word);
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
