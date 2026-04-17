use std::path::{Path, PathBuf};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tiktag::{Tiktag, missing_model_files};

const PROFILES_PATH: &str = "models/profiles.toml";
const MODEL_DIR: &str = "models/distilbert-base-multilingual-cased-ner-hrl";

fn manifest_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn assets_present() -> bool {
    missing_model_files(&manifest_path(MODEL_DIR)).is_empty()
}

fn corpora() -> Vec<(&'static str, String)> {
    vec![
        (
            "short",
            "Satya Nadella met Sundar Pichai in Paris.".to_owned(),
        ),
        (
            "medium",
            concat!(
                "Maria Garcia works at OpenAI in San Francisco. ",
                "She collaborates with Yann LeCun on research at Meta. ",
                "Last Tuesday, Satya Nadella announced a new partnership with Anthropic in London. ",
                "The team at DeepMind in Mountain View is also contributing.",
            )
            .to_owned(),
        ),
        ("long", {
            // Roughly 2x max_tokens to exercise the sliding window path.
            let base = "Dr. Alice Chen presented at MIT alongside John Smith from Stanford. \
                        The panel discussed AI safety at the Brookings Institution in Washington. \
                        Later, Emma Watson joined from the BBC office in London to debate Elon Musk. \
                        Meanwhile, Jensen Huang at NVIDIA in Santa Clara demonstrated new GPUs. ";
            base.repeat(12)
        }),
    ]
}

fn bench_anonymize(c: &mut Criterion) {
    if !assets_present() {
        eprintln!("tiktag bench: model assets missing at {MODEL_DIR}; skipping.");
        return;
    }

    let mut tiktag = Tiktag::new(&manifest_path(PROFILES_PATH))
        .expect("Tiktag::new should succeed when model assets are present");

    let mut group = c.benchmark_group("anonymize");
    for (name, text) in corpora() {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, text| {
            b.iter(|| {
                tiktag
                    .anonymize(text)
                    .expect("anonymize should succeed on bench corpus")
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_anonymize);
criterion_main!(benches);
