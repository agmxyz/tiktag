use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "lil-inference",
    about = "Developer-first ONNX NER/PII inference CLI"
)]
pub struct Args {
    #[arg(long, default_value = "models/profiles.toml")]
    pub profiles: PathBuf,

    #[arg(long)]
    pub profile: Option<String>,

    #[arg(long, default_value_t = false)]
    pub show_tokens: bool,

    #[arg(long, default_value_t = false)]
    pub json: bool,

    pub text: String,
}
