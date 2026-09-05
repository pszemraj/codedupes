use cowsay_dupe_fixture::{render, CowOptions, WrapAlgorithm};

const SPEECH: &str = include_str!("../src/bubble/speech.rs");
const THOUGHT: &str = include_str!("../src/bubble/thought.rs");
const SCANNER: &str = include_str!("../src/wrapping/scanner.rs");
const FOLD: &str = include_str!("../src/wrapping/fold.rs");

#[test]
fn exact_clone_is_still_byte_identical() {
    let speech = region(
        SPEECH,
        "// fixture:exact-border:start",
        "// fixture:exact-border:end",
    );
    let thought = region(
        THOUGHT,
        "// fixture:exact-border:start",
        "// fixture:exact-border:end",
    );

    assert_eq!(speech, thought, "the planted exact clone drifted");
}

#[test]
fn edit_distance_clone_is_similar_but_not_exact() {
    let speech = normalized(region(
        SPEECH,
        "// fixture:edit-bubble:start",
        "// fixture:edit-bubble:end",
    ));
    let thought = normalized(region(
        THOUGHT,
        "// fixture:edit-bubble:start",
        "// fixture:edit-bubble:end",
    ));

    assert_ne!(speech, thought, "the near clone became exact");
    let similarity = levenshtein_similarity(&speech, &thought);
    assert!(
        (0.45..0.98).contains(&similarity),
        "unexpected edit-clone similarity: {similarity:.3}"
    );
}

#[test]
fn semantic_clone_has_different_source_but_equal_behavior() {
    let scanner_source = normalized(region(
        SCANNER,
        "// fixture:semantic-wrap:start",
        "// fixture:semantic-wrap:end",
    ));
    let fold_source = normalized(region(
        FOLD,
        "// fixture:semantic-wrap:start",
        "// fixture:semantic-wrap:end",
    ));
    assert_ne!(scanner_source, fold_source);

    for message in [
        "a compact fixture",
        "one two three four five six",
        "hard-break-this-unbroken-token",
        "first paragraph\n\nthird paragraph",
    ] {
        for width in 4..=20 {
            let scanner = render(
                message,
                CowOptions {
                    width,
                    thinking: false,
                    wrap_algorithm: WrapAlgorithm::Scanner,
                },
            );
            let fold = render(
                message,
                CowOptions {
                    width,
                    thinking: false,
                    wrap_algorithm: WrapAlgorithm::Fold,
                },
            );
            assert_eq!(scanner, fold, "message={message:?}, width={width}");
        }
    }
}

fn region<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = source.find(start_marker).expect("missing start marker");
    let end_offset = source[start..]
        .find(end_marker)
        .expect("missing end marker");
    let end = start + end_offset + end_marker.len();
    &source[start..end]
}

fn normalized(source: &str) -> String {
    source.split_whitespace().collect()
}

fn levenshtein_similarity(left: &str, right: &str) -> f64 {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let denominator = left.len().max(right.len());
    if denominator == 0 {
        return 1.0;
    }

    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];

    for (left_index, left_character) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_character) in right.iter().enumerate() {
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            let substitution = previous[right_index]
                + if left_character == right_character {
                    0
                } else {
                    1
                };
            current[right_index + 1] = insertion.min(deletion).min(substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    1.0 - previous[right.len()] as f64 / denominator as f64
}
