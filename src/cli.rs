// CLI argument definitions.
//
// `tiktag "text"` remains the default run path.
// `tiktag download` fetches the built-in eu-pii model assets.

use std::ffi::OsString;

use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "tiktag",
    about = "eu-pii text anonymization CLI",
    after_help = "Download model assets with `tiktag download`.\nUse `--stdin` to read input text from standard input.\nUse `tiktag -- download` to run anonymization on the literal text `download`."
)]
pub struct RunArgs {
    /// Print per-token predictions to stderr (debug aid).
    #[arg(long, default_value_t = false)]
    pub show_tokens: bool,

    /// Read input text from stdin instead of a positional argument.
    #[arg(long, default_value_t = false, conflicts_with = "text")]
    pub stdin: bool,

    /// Emit JSON output instead of human-readable lines.
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// The text to run NER/PII inference on (omit when using --stdin).
    #[arg(required_unless_present = "stdin")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Parser)]
#[command(name = "tiktag download", about = "Download built-in model assets")]
pub struct DownloadArgs {}

#[derive(Debug, Clone)]
pub enum Command {
    Run(RunArgs),
    Download(DownloadArgs),
}

pub fn parse() -> Command {
    parse_from(std::env::args_os())
}

fn parse_from<I, T>(args: I) -> Command
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let args = args.into_iter().map(Into::into).collect::<Vec<_>>();
    let program_name = args
        .first()
        .cloned()
        .unwrap_or_else(|| OsString::from("tiktag"));

    if matches!(args.get(1).and_then(|arg| arg.to_str()), Some("download")) {
        let forwarded = std::iter::once(program_name).chain(args.into_iter().skip(2));
        Command::Download(DownloadArgs::parse_from(forwarded))
    } else {
        Command::Run(RunArgs::parse_from(args))
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, parse_from};

    #[test]
    fn parses_run_args_by_default() {
        let command = parse_from(["tiktag", "hello world"]);

        let Command::Run(args) = command else {
            panic!("expected run command");
        };
        assert_eq!(args.text.as_deref(), Some("hello world"));
        assert!(!args.stdin);
    }

    #[test]
    fn parses_download_command() {
        let command = parse_from(["tiktag", "download"]);

        let Command::Download(_) = command else {
            panic!("expected download command");
        };
    }

    #[test]
    fn allows_literal_download_text_with_double_dash() {
        let command = parse_from(["tiktag", "--", "download"]);

        let Command::Run(args) = command else {
            panic!("expected run command");
        };
        assert_eq!(args.text.as_deref(), Some("download"));
        assert!(!args.stdin);
    }

    #[test]
    fn parses_stdin_mode_without_text() {
        let command = parse_from(["tiktag", "--stdin", "--json"]);

        let Command::Run(args) = command else {
            panic!("expected run command");
        };
        assert!(args.stdin);
        assert!(args.text.is_none());
        assert!(args.json);
    }
}
