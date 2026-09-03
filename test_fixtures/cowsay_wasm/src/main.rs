use std::env;
use std::io::{self, Read};
use std::process;

use cowsay_dupe_fixture::{render, CowOptions, WrapAlgorithm};

fn main() {
    match run() {
        Ok(rendered) => println!("{rendered}"),
        Err(error) => {
            eprintln!("error: {error}\n");
            print_usage();
            process::exit(2);
        }
    }
}

fn run() -> Result<String, String> {
    let mut args = env::args().skip(1).peekable();
    let mut options = CowOptions::default();
    let mut words = Vec::new();

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--think" | "-t" => options.thinking = true,
            "--width" | "-w" => {
                let raw_width = args
                    .next()
                    .ok_or_else(|| "--width requires an integer".to_owned())?;
                options.width = raw_width
                    .parse::<usize>()
                    .map_err(|_| format!("invalid width: {raw_width}"))?;
            }
            "--wrapper" => {
                let wrapper = args
                    .next()
                    .ok_or_else(|| "--wrapper requires scanner or fold".to_owned())?;
                options.wrap_algorithm = match wrapper.as_str() {
                    "scanner" => WrapAlgorithm::Scanner,
                    "fold" => WrapAlgorithm::Fold,
                    _ => return Err(format!("unknown wrapper: {wrapper}")),
                };
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            "--" => {
                words.extend(args);
                break;
            }
            _ if argument.starts_with('-') => {
                return Err(format!("unknown option: {argument}"));
            }
            _ => words.push(argument),
        }
    }

    let message = if words.is_empty() {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|error| format!("failed to read stdin: {error}"))?;
        input
            .trim_end_matches(|character| character == '\n' || character == '\r')
            .to_owned()
    } else {
        words.join(" ")
    };

    Ok(render(&message, options))
}

fn print_usage() {
    eprintln!(
        "cowsay-fixture [OPTIONS] [MESSAGE...]\n\n\
         Options:\n\
           -t, --think              use a thought bubble\n\
           -w, --width <COLUMNS>    wrap at 4..=96 columns (default: 40)\n\
               --wrapper <NAME>     scanner or fold (default: scanner)\n\
           -h, --help               show this help\n\n\
         With no MESSAGE, input is read from stdin."
    );
}
