use super::shared::{display_width, split_word};

// fixture:semantic-wrap:start
pub(crate) fn wrap(message: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let wrapped: Vec<String> = message
        .split('\n')
        .flat_map(|raw_paragraph| {
            let paragraph = raw_paragraph.strip_suffix('\r').unwrap_or(raw_paragraph);
            let pieces: Vec<String> = paragraph
                .split_whitespace()
                .flat_map(|word| split_word(word, width))
                .collect();

            if pieces.is_empty() {
                return vec![String::new()];
            }

            let (mut complete, active, _) = pieces.into_iter().fold(
                (Vec::new(), String::new(), 0usize),
                |(mut complete, mut active, active_width), piece| {
                    let piece_width = display_width(&piece);
                    match (
                        active.is_empty(),
                        active_width + 1 + piece_width <= width,
                    ) {
                        (true, _) => (complete, piece, piece_width),
                        (false, true) => {
                            active.push(' ');
                            active.push_str(&piece);
                            (complete, active, active_width + 1 + piece_width)
                        }
                        (false, false) => {
                            complete.push(active);
                            (complete, piece, piece_width)
                        }
                    }
                },
            );

            if !active.is_empty() {
                complete.push(active);
            }
            complete
        })
        .collect();

    if wrapped.is_empty() {
        vec![String::new()]
    } else {
        wrapped
    }
}
// fixture:semantic-wrap:end
