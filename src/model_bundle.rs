use std::path::{Path, PathBuf};

use anyhow::bail;

pub const REQUIRED_MODEL_FILES: &[&str] =
    &["tokenizer.json", "config.json", "onnx/model_quantized.onnx"];

pub fn missing_model_files(model_dir: &Path) -> Vec<PathBuf> {
    REQUIRED_MODEL_FILES
        .iter()
        .map(|relative_path| model_dir.join(relative_path))
        .filter(|path| !path.is_file())
        .collect()
}

pub fn validate_model_bundle(model_dir: &Path) -> anyhow::Result<()> {
    let missing = missing_model_files(model_dir);
    if missing.is_empty() {
        return Ok(());
    }

    let missing_list = missing
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    bail!(
        "model dir '{}' is missing required files: {}",
        model_dir.display(),
        missing_list
    );
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::validate_model_bundle;

    #[test]
    fn rejects_missing_required_model_files() {
        let temp_dir = tempdir().expect("temp dir");
        let err = validate_model_bundle(temp_dir.path()).expect_err("missing files should fail");
        let message = err.to_string();

        assert!(message.contains("tokenizer.json"));
        assert!(message.contains("config.json"));
        assert!(message.contains("onnx/model_quantized.onnx"));
    }

    #[test]
    fn accepts_valid_model_bundle_shape() {
        let temp_dir = tempdir().expect("temp dir");
        fs::write(temp_dir.path().join("tokenizer.json"), "{}").expect("tokenizer");
        fs::write(temp_dir.path().join("config.json"), "{}").expect("config");
        fs::create_dir_all(temp_dir.path().join("onnx")).expect("onnx dir");
        fs::write(temp_dir.path().join("onnx/model_quantized.onnx"), "").expect("onnx model");

        validate_model_bundle(temp_dir.path()).expect("bundle should validate");
    }
}
