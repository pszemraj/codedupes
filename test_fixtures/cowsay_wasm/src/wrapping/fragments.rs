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
