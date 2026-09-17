/// Partition one nonempty token into Unicode-scalar fragments of a fixed size.
pub(crate) fn segment(token: &str, chunk_length: usize) -> Vec<String> {
    let chunk_length = chunk_length.max(1);
    token.chars().enumerate().fold(
        Vec::new(),
        |mut fragments: Vec<String>, (offset, symbol)| {
            if offset % chunk_length == 0 {
                fragments.push(String::new());
            }
            fragments
                .last_mut()
                .expect("a fragment is opened before its symbol")
                .push(symbol);
            fragments
        },
    )
}

/// Partition a token by slicing a collected scalar buffer at indexed boundaries.
pub(crate) fn segment_indexed(token: &str, chunk_length: usize) -> Vec<String> {
    let chunk_length = chunk_length.max(1);
    let characters: Vec<char> = token.chars().collect();
    let mut fragments = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let end = (start + chunk_length).min(characters.len());
        fragments.push(characters[start..end].iter().collect());
        start = end;
    }
    fragments
}

/// Partition a token through recursive consumption of a scalar buffer.
pub(crate) fn segment_recursive(token: &str, chunk_length: usize) -> Vec<String> {
    fn collect_chunks(characters: &[char], width: usize, output: &mut Vec<String>) {
        if characters.is_empty() {
            return;
        }
        let boundary = width.min(characters.len());
        output.push(characters[..boundary].iter().collect());
        collect_chunks(&characters[boundary..], width, output);
    }

    let characters: Vec<char> = token.chars().collect();
    let mut fragments = Vec::new();
    collect_chunks(&characters, chunk_length.max(1), &mut fragments);
    fragments
}

#[cfg(test)]
mod tests {
    use super::{segment, segment_indexed, segment_recursive};

    #[test]
    fn segment_variants_preserve_unicode_scalar_boundaries() {
        for token in ["a", "abcdefghij", "naïve", "東京café"] {
            for width in 1..=8 {
                let expected = segment(token, width);
                assert_eq!(segment_indexed(token, width), expected);
                assert_eq!(segment_recursive(token, width), expected);
            }
        }
    }
}
