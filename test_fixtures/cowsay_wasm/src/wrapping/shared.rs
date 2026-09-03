pub(crate) fn display_width(input: &str) -> usize {
    input.chars().count()
}

pub(crate) fn split_word(word: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    if display_width(word) <= width {
        return vec![word.to_owned()];
    }

    let mut pieces = Vec::new();
    let mut piece = String::new();
    let mut piece_width = 0;

    for character in word.chars() {
        if piece_width == width {
            pieces.push(std::mem::take(&mut piece));
            piece_width = 0;
        }

        piece.push(character);
        piece_width += 1;
    }

    if !piece.is_empty() {
        pieces.push(piece);
    }

    pieces
}
