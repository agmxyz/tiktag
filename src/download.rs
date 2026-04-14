use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, bail};
use log::info;

use crate::model_bundle::validate_model_bundle;
use crate::profiles::{Profiles, ResolvedProfile};

pub fn download() -> anyhow::Result<()> {
    download_with_program_from_profiles(
        Path::new(crate::profiles::INTERNAL_PROFILES_PATH),
        Path::new("hf"),
    )
}

fn download_with_program_from_profiles(profiles_path: &Path, program: &Path) -> anyhow::Result<()> {
    let profiles = Profiles::load(profiles_path)
        .with_context(|| format!("failed to load profiles from {}", profiles_path.display()))?;
    let resolved = profiles.resolve_default();
    download_profile_with_program(&resolved, program)
}

fn download_profile_with_program(profile: &ResolvedProfile, program: &Path) -> anyhow::Result<()> {
    info!(
        "downloading assets for model='{}' repo='{}' into '{}'",
        profile.name,
        profile.hf_repo,
        profile.model_dir.display()
    );

    fs::create_dir_all(&profile.model_dir)
        .with_context(|| format!("failed to create model dir {}", profile.model_dir.display()))?;

    let status = Command::new(program)
        .arg("download")
        .arg(&profile.hf_repo)
        .arg("config.json")
        .arg("tokenizer.json")
        .arg("onnx/model_quantized.onnx")
        .arg("--local-dir")
        .arg(&profile.model_dir)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status();

    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            bail!(
                "hf download failed for model '{}' repo '{}' into '{}' with status {}",
                profile.name,
                profile.hf_repo,
                profile.model_dir.display(),
                status
            );
        }
        Err(err) if err.kind() == ErrorKind::NotFound => {
            bail!(
                "failed to run `hf` while downloading model '{}': command not found. Install the Hugging Face CLI and ensure `hf` is on PATH.",
                profile.name
            );
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "failed to run `hf download` for model '{}' repo '{}' into '{}'",
                    profile.name,
                    profile.hf_repo,
                    profile.model_dir.display()
                )
            });
        }
    }

    validate_model_bundle(&profile.model_dir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};

    use tempfile::tempdir;

    use super::download_with_program_from_profiles;

    fn write_profiles_file(base_dir: &Path, model_dir: &str) -> PathBuf {
        let path = base_dir.join("profiles.toml");
        fs::write(
            &path,
            format!(
                r#"
default_profile = "distilbert_ner_hrl"

[profiles.distilbert_ner_hrl]
hf_repo = "example/default"
model_dir = "{model_dir}"
max_tokens = 512
overlap_tokens = 128
"#
            ),
        )
        .expect("profiles file");
        path
    }

    fn write_fake_hf_script(dir: &Path, body: &str) -> PathBuf {
        let script_path = dir.join("fake-hf.sh");
        fs::write(&script_path, body).expect("script");
        let mut permissions = fs::metadata(&script_path).expect("metadata").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script_path, permissions).expect("permissions");
        script_path
    }

    #[test]
    fn downloads_builtin_model_with_expected_arguments() {
        let temp_dir = tempdir().expect("temp dir");
        let profiles_path = write_profiles_file(temp_dir.path(), "default-model");
        let args_path = temp_dir.path().join("hf-args.txt");
        let script_path = write_fake_hf_script(
            temp_dir.path(),
            &format!(
                "#!/bin/sh\nset -eu\nprintf '%s\n' \"$@\" > \"{}\"\ndest=''\nprev=''\nfor arg in \"$@\"; do\n  if [ \"$prev\" = \"--local-dir\" ]; then\n    dest=\"$arg\"\n  fi\n  prev=\"$arg\"\ndone\nmkdir -p \"$dest/onnx\"\n: > \"$dest/config.json\"\n: > \"$dest/tokenizer.json\"\n: > \"$dest/onnx/model_quantized.onnx\"\n",
                args_path.display()
            ),
        );

        download_with_program_from_profiles(&profiles_path, &script_path)
            .expect("download should succeed");

        let args = fs::read_to_string(&args_path).expect("args");
        assert_eq!(
            args.lines().collect::<Vec<_>>(),
            vec![
                "download",
                "example/default",
                "config.json",
                "tokenizer.json",
                "onnx/model_quantized.onnx",
                "--local-dir",
                temp_dir
                    .path()
                    .join("default-model")
                    .to_string_lossy()
                    .as_ref(),
            ]
        );
    }

    #[test]
    fn loads_default_profile_when_downloading() {
        let temp_dir = tempdir().expect("temp dir");
        let profiles_path = write_profiles_file(temp_dir.path(), "default-model");
        let args_path = temp_dir.path().join("hf-args.txt");
        let script_path = write_fake_hf_script(
            temp_dir.path(),
            &format!(
                "#!/bin/sh\nset -eu\nprintf '%s\n' \"$@\" > \"{}\"\ndest=''\nprev=''\nfor arg in \"$@\"; do\n  if [ \"$prev\" = \"--local-dir\" ]; then\n    dest=\"$arg\"\n  fi\n  prev=\"$arg\"\ndone\nmkdir -p \"$dest/onnx\"\n: > \"$dest/config.json\"\n: > \"$dest/tokenizer.json\"\n: > \"$dest/onnx/model_quantized.onnx\"\n",
                args_path.display()
            ),
        );

        download_with_program_from_profiles(&profiles_path, &script_path)
            .expect("default download");

        let args = fs::read_to_string(&args_path).expect("args");
        assert_eq!(args.lines().collect::<Vec<_>>()[1], "example/default");
    }

    #[test]
    fn resolves_absolute_model_dir() {
        let temp_dir = tempdir().expect("temp dir");
        let absolute_dir = temp_dir.path().join("absolute-model");
        let profiles_path =
            write_profiles_file(temp_dir.path(), absolute_dir.to_string_lossy().as_ref());
        let args_path = temp_dir.path().join("hf-args.txt");
        let script_path = write_fake_hf_script(
            temp_dir.path(),
            &format!(
                "#!/bin/sh\nset -eu\nprintf '%s\n' \"$@\" > \"{}\"\ndest=''\nprev=''\nfor arg in \"$@\"; do\n  if [ \"$prev\" = \"--local-dir\" ]; then\n    dest=\"$arg\"\n  fi\n  prev=\"$arg\"\ndone\nmkdir -p \"$dest/onnx\"\n: > \"$dest/config.json\"\n: > \"$dest/tokenizer.json\"\n: > \"$dest/onnx/model_quantized.onnx\"\n",
                args_path.display()
            ),
        );

        download_with_program_from_profiles(&profiles_path, &script_path)
            .expect("absolute download");

        let args = fs::read_to_string(&args_path).expect("args");
        assert_eq!(
            args.lines().last().expect("dest"),
            absolute_dir.to_string_lossy().as_ref()
        );
    }

    #[test]
    fn surfaces_missing_hf_binary() {
        let temp_dir = tempdir().expect("temp dir");
        let profiles_path = write_profiles_file(temp_dir.path(), "default-model");
        let missing_binary = temp_dir.path().join("missing-hf");

        let err = download_with_program_from_profiles(&profiles_path, &missing_binary)
            .expect_err("missing binary should fail");

        assert!(err.to_string().contains("Install the Hugging Face CLI"));
    }

    #[test]
    fn surfaces_non_zero_hf_exit() {
        let temp_dir = tempdir().expect("temp dir");
        let profiles_path = write_profiles_file(temp_dir.path(), "default-model");
        let script_path = write_fake_hf_script(temp_dir.path(), "#!/bin/sh\nexit 17\n");

        let err = download_with_program_from_profiles(&profiles_path, &script_path)
            .expect_err("non-zero exit should fail");

        assert!(err.to_string().contains("status exit status: 17"));
    }

    #[test]
    fn rejects_incomplete_bundle_after_successful_download() {
        let temp_dir = tempdir().expect("temp dir");
        let profiles_path = write_profiles_file(temp_dir.path(), "default-model");
        let script_path = write_fake_hf_script(
            temp_dir.path(),
            "#!/bin/sh\nset -eu\ndest=''\nprev=''\nfor arg in \"$@\"; do\n  if [ \"$prev\" = \"--local-dir\" ]; then\n    dest=\"$arg\"\n  fi\n  prev=\"$arg\"\ndone\nmkdir -p \"$dest\"\n: > \"$dest/config.json\"\n",
        );

        let err = download_with_program_from_profiles(&profiles_path, &script_path)
            .expect_err("incomplete bundle should fail");

        let message = err.to_string();
        assert!(message.contains("tokenizer.json"));
        assert!(message.contains("onnx/model_quantized.onnx"));
    }
}
