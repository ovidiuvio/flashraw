//! `Demosaic::EdgeAware` on real files: the same bytes for any thread count, backend and
//! output encoding path, a different (cleaner) image than bilinear, and the frame geometry
//! of the other demosaics.
//!
//! Runs when the CC0 corpus is present (`$FLASHRAW_CORPUS` or `<repo>/corpus`, fetched with
//! `tools/corpus/fetch.py`); skipped otherwise.

use flashraw::{Backend, DecodeOptions, Demosaic, FlashRaw, OutputFormat, Rect};
use std::path::PathBuf;

/// A Bayer and an X-Trans file of the corpus.
const FILES: &[&str] = &["Canon/EOS 40D/_MG_0153.CR2", "Fujifilm/X-T5/DSCF0021.RAF"];

fn corpus() -> Option<PathBuf> {
    let c = std::env::var("FLASHRAW_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus"));
    c.exists().then_some(c)
}

fn opts(output: OutputFormat, demosaic: Demosaic) -> DecodeOptions {
    DecodeOptions { output, demosaic, ..Default::default() }
}

#[test]
fn edge_aware_is_deterministic_and_differs_from_bilinear() {
    let Some(corpus) = corpus() else {
        eprintln!("corpus not found; skipping (see tools/corpus/fetch.py)");
        return;
    };
    let mut ran = 0;
    for rel in FILES {
        let path = corpus.join(rel);
        if !path.exists() {
            eprintln!("missing {rel}");
            continue;
        }
        ran += 1;
        let raw = FlashRaw::open_path(&path).unwrap();
        let region = Rect { x: 400, y: 300, width: 700, height: 500 };
        for output in [OutputFormat::Rgba8, OutputFormat::LinearRgbF32] {
            let base = DecodeOptions { region: Some(region), ..opts(output, Demosaic::EdgeAware) };
            let reference =
                raw.decode(DecodeOptions { threads: Some(1), backend: Backend::Cpu, ..base.clone() }).unwrap();
            for (threads, backend) in [(Some(3), Backend::Cpu), (None, Backend::CpuSimd), (None, Backend::Auto)] {
                let again = raw.decode(DecodeOptions { threads, backend, ..base.clone() }).unwrap();
                assert!(again.data == reference.data, "{rel} {output:?}: {threads:?} {backend:?} differs");
            }
            let bilinear = raw.decode(DecodeOptions { demosaic: Demosaic::Bilinear, ..base.clone() }).unwrap();
            assert_eq!((bilinear.width(), bilinear.height()), (reference.width(), reference.height()), "{rel}");
            assert!(bilinear.data != reference.data, "{rel}: edge-aware equals bilinear");
        }
        // Whole frame: same geometry as bilinear, no warnings.
        let full = raw.decode(opts(OutputFormat::Rgba8, Demosaic::EdgeAware)).unwrap();
        let bil = raw.decode(opts(OutputFormat::Rgba8, Demosaic::Bilinear)).unwrap();
        assert_eq!((full.width(), full.height()), (bil.width(), bil.height()), "{rel}");
        assert!(full.warnings().is_empty(), "{rel}");
        // Close to bilinear overall (same colours), and not a copy: mean absolute difference.
        let (a, b) = (full.as_u8().unwrap(), bil.as_u8().unwrap());
        let mad = a.iter().zip(b).map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64).sum::<u64>() as f64
            / a.len() as f64;
        assert!(mad > 0.05 && mad < 12.0, "{rel}: mean absolute difference to bilinear {mad}");
    }
    if ran == 0 {
        eprintln!("none of the files is in the corpus; skipping");
    }
}

#[test]
fn edge_aware_gpu_requests_fall_back_to_the_cpu() {
    let Some(corpus) = corpus() else { return };
    let path = corpus.join(FILES[0]);
    if !path.exists() {
        return;
    }
    let raw = FlashRaw::open_path(&path).unwrap();
    let cpu = raw.decode(DecodeOptions { backend: Backend::Cpu, ..opts(OutputFormat::Rgba8, Demosaic::EdgeAware) });
    let gpu = raw.decode(DecodeOptions { backend: Backend::Auto, ..opts(OutputFormat::Rgba8, Demosaic::EdgeAware) });
    assert!(cpu.unwrap().data == gpu.unwrap().data);
    // GPU-resident outputs cannot fall back.
    let r = raw.decode(opts(OutputFormat::GpuRgba8, Demosaic::EdgeAware));
    assert!(r.is_err());
}
