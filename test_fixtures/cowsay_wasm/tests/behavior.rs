use cowsay_dupe_fixture::{cowsay, render, CowOptions, WrapAlgorithm};

#[test]
fn renders_single_line_speech() {
    let output = cowsay("moo", 40, false, false);

    assert!(output.starts_with(" _____\n< moo >\n -----\n"));
    assert!(output.contains("\\   ^__^"));
}

#[test]
fn renders_thought_mode() {
    let output = cowsay("consider the borrow checker", 40, true, false);

    assert!(output.contains("( consider the borrow checker )"));
    assert!(output.contains("o   ^__^"));
}

#[test]
fn wraps_long_words_without_losing_characters() {
    let output = cowsay("abcdefghij", 4, false, false);

    assert!(output.contains("/ abcd \\"));
    assert!(output.contains("| efgh |"));
    assert!(output.contains("\\ ij   /"));
}

#[test]
fn both_semantic_wrappers_render_identically() {
    let corpus = [
        "",
        "moo",
        "the quick brown fox jumps over the lazy cow",
        "one\n\nthree",
        "supercalifragilisticexpialidocious",
        "naïve café 東京",
        "spaces    are\tcollapsed",
        "windows\r\nline endings",
    ];

    for message in corpus {
        for width in [4, 7, 12, 40, 96] {
            for thinking in [false, true] {
                let scanner = render(
                    message,
                    CowOptions {
                        width,
                        thinking,
                        wrap_algorithm: WrapAlgorithm::Scanner,
                    },
                );
                let fold = render(
                    message,
                    CowOptions {
                        width,
                        thinking,
                        wrap_algorithm: WrapAlgorithm::Fold,
                    },
                );

                assert_eq!(scanner, fold, "message={message:?}, width={width}");
            }
        }
    }
}

#[test]
fn public_width_clamps_and_unicode_scalar_boundaries() {
    for fold in [false, true] {
        assert_eq!(
            cowsay("abcdefghij", 0, false, fold),
            cowsay("abcdefghij", 4, false, fold)
        );
        assert_eq!(
            cowsay("moo", 1000, false, fold),
            cowsay("moo", 96, false, fold)
        );
        let rendered = cowsay("東京 café", 4, false, fold);
        assert!(rendered.starts_with(" ______\n/ 東京   \\\n\\ café /\n ------\n"));
        assert_ne!(cowsay("moo", 4, false, fold), cowsay("moo", 4, true, fold));
    }
}

#[test]
fn native_cli_reports_invalid_options_and_runs_both_wrappers() {
    use std::process::Command;
    let executable = env!("CARGO_BIN_EXE_cowsay-fixture");
    let invalid = Command::new(executable)
        .args(["--wrapper", "unknown"])
        .output()
        .unwrap();
    assert_eq!(invalid.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("unknown wrapper"));
    for wrapper in ["scanner", "fold"] {
        let output = Command::new(executable)
            .args(["--wrapper", wrapper, "--width", "4", "東京 café"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("café"));
    }
}
