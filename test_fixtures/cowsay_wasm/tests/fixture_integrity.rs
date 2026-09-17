use cowsay_dupe_fixture::{render, CowOptions, WrapAlgorithm};

const SPEECH: &str = include_str!("../src/bubble/speech.rs");
const THOUGHT: &str = include_str!("../src/bubble/thought.rs");
const SCANNER: &str = include_str!("../src/wrapping/scanner.rs");
const CURSOR: &str = include_str!("../src/wrapping/cursor.rs");
const FOLD: &str = include_str!("../src/wrapping/fold.rs");
const FRAGMENTS: &str = include_str!("../src/wrapping/fragments.rs");
const QUEUE: &str = include_str!("../src/wrapping/queue.rs");
const SHARED: &str = include_str!("../src/wrapping/shared.rs");

#[test]
fn exact_clone_is_still_byte_identical() {
    let speech = function(SPEECH, "pub(crate) fn make_borders");
    let thought = function(THOUGHT, "pub(crate) fn make_borders");

    assert_eq!(speech, thought, "the planted exact clone drifted");
}

#[test]
fn edited_clone_remains_distinct() {
    let speech = normalized(function(SPEECH, "pub(crate) fn render_bubble"));
    let thought = normalized(function(THOUGHT, "pub(crate) fn render_bubble"));

    assert_ne!(speech, thought, "the near clone became exact");
}

#[test]
fn semantic_clone_has_different_source_but_equal_behavior() {
    let scanner_source = normalized(function(SCANNER, "pub(crate) fn wrap"));
    let fold_source = normalized(function(FOLD, "pub(crate) fn wrap"));
    let queue_source = normalized(function(QUEUE, "pub(crate) fn wrap"));
    let cursor_source = normalized(function(CURSOR, "pub(crate) fn wrap"));
    let shared_splitter = normalized(function(SHARED, "pub(crate) fn split_word"));
    let queue_fragmenter = normalized(function(FRAGMENTS, "pub(crate) fn segment"));
    assert_ne!(scanner_source, fold_source);
    assert_ne!(scanner_source, queue_source);
    assert_ne!(fold_source, queue_source);
    assert_ne!(scanner_source, cursor_source);
    assert_ne!(fold_source, cursor_source);
    assert_ne!(queue_source, cursor_source);
    assert_ne!(shared_splitter, queue_fragmenter);

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
                    render_algorithm: cowsay_dupe_fixture::RenderAlgorithm::Pipeline,
                },
            );
            let fold = render(
                message,
                CowOptions {
                    width,
                    thinking: false,
                    wrap_algorithm: WrapAlgorithm::Fold,
                    render_algorithm: cowsay_dupe_fixture::RenderAlgorithm::Pipeline,
                },
            );
            let queue = render(
                message,
                CowOptions {
                    width,
                    thinking: false,
                    wrap_algorithm: WrapAlgorithm::Queue,
                    render_algorithm: cowsay_dupe_fixture::RenderAlgorithm::Pipeline,
                },
            );
            assert_eq!(scanner, fold, "message={message:?}, width={width}");
            assert_eq!(scanner, queue, "message={message:?}, width={width}");
        }
    }
}

fn function<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source.find(signature).expect("missing function signature");
    let body = source[start..].find('{').expect("missing function body") + start;
    let mut depth = 0;
    for (offset, character) in source[body..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..body + offset + 1];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function body")
}

fn normalized(source: &str) -> String {
    source.split_whitespace().collect()
}
