use std::path::PathBuf;

/// Public error type for the tiktag library surface.
#[derive(Debug, thiserror::Error)]
pub enum TiktagError {
    /// Profile file could not be read from disk.
    #[error("failed to read profile file {path}: {source}")]
    ProfileRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Profile file is not valid TOML or has wrong shape.
    #[error("failed to parse profile file {path}: {source}")]
    ProfileParse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    /// Profile passed semantic validation but contained invalid values.
    #[error("invalid profile: {0}")]
    ProfileInvalid(String),

    /// Required model bundle files are missing from the model directory.
    #[error("model dir '{path}' is missing required files: {missing:?}")]
    ModelBundleMissing {
        path: PathBuf,
        missing: Vec<PathBuf>,
    },

    /// Tokenizer failed to load.
    #[error("failed to load tokenizer: {0}")]
    Tokenizer(String),

    /// Model config.json could not be parsed.
    #[error("failed to parse config.json: {0}")]
    Config(String),

    /// ONNX runtime reported an error.
    #[error("ONNX runtime error: {0}")]
    OrtRuntime(String),

    /// Generic I/O failure tied to a path.
    #[error("I/O error: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Escape hatch for not-yet-typed internal errors. Use sparingly.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Library-wide `Result` alias that defaults to `TiktagError`.
pub type Result<T> = std::result::Result<T, TiktagError>;
