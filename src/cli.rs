// CLI argument definitions. All flags are optional; only `text` is required.

use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "tiktag", about = "ONNX NER/PII inference CLI")]
pub struct Args {
    /// Path to profiles TOML file.
    #[arg(long, default_value = "models/profiles.toml")]
    pub profiles: PathBuf,

    /// Which profile to use. Falls back to the file's default_profile.
    #[arg(long)]
    pub profile: Option<String>,

    /// Print per-token predictions to stderr (debug aid).
    #[arg(long, default_value_t = false)]
    pub show_tokens: bool,

    /// Emit JSON output instead of human-readable lines.
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// The text to run NER/PII inference on.
    pub text: String,
}
