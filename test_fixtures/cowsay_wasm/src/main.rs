use std::env;
use std::io::{self, Read};
use std::process;

use cowsay_dupe_fixture::{render, scheduler, CowOptions, RenderAlgorithm, WrapAlgorithm};

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
            "--scheduler-demo" => {
                if args.next().is_some() {
                    return Err(
                        "--scheduler-demo does not accept a message or other options".to_owned(),
                    );
                }
                return scheduler::demo_selection()
                    .map(|ids| format!("Selected job ids: {ids:?}"))
                    .map_err(|error| format!("scheduler demo failed: {error:?}"));
            }
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
                let wrapper = args.next().ok_or_else(|| {
                    "--wrapper requires scanner, fold, queue, or cursor".to_owned()
                })?;
                options.wrap_algorithm = match wrapper.as_str() {
                    "scanner" => WrapAlgorithm::Scanner,
                    "fold" => WrapAlgorithm::Fold,
                    "queue" => WrapAlgorithm::Queue,
                    "cursor" => WrapAlgorithm::Cursor,
                    _ => return Err(format!("unknown wrapper: {wrapper}")),
                };
            }
            "--renderer" => {
                let renderer = args
                    .next()
                    .ok_or_else(|| "--renderer requires pipeline or composed".to_owned())?;
                options.render_algorithm = match renderer.as_str() {
                    "pipeline" => RenderAlgorithm::Pipeline,
                    "composed" => RenderAlgorithm::Composed,
                    _ => return Err(format!("unknown renderer: {renderer}")),
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
        input.trim_end_matches(['\n', '\r']).to_owned()
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
           --wrapper <NAME>     scanner, fold, queue, or cursor (default: scanner)\n\
           --renderer <NAME>    pipeline or composed (default: pipeline)\n\
           --scheduler-demo      run the offline scheduler selection example\n\
           -h, --help               show this help\n\n\
         With no MESSAGE, input is read from stdin."
    );
}
