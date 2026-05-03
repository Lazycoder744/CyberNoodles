use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use bsor_tools::builder::{build_bc_dataset, BuildBcDatasetArgs};
use bsor_tools::io::{read_bsor_path, write_bsor_path};
use bsor_tools::sanitize::{dataset_view, validation_summary};

#[derive(Parser)]
#[command(name = "bsor_tools")]
#[command(about = "Rust BSOR parser, writer, and audit utilities for CyberNoodles")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    DumpJson {
        input: PathBuf,
        #[arg(long)]
        pretty: bool,
    },
    DumpDatasetJson {
        input: PathBuf,
        #[arg(long)]
        pretty: bool,
    },
    WriteJson {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        input_json: Option<PathBuf>,
    },
    Validate {
        input: PathBuf,
        #[arg(long)]
        pretty: bool,
    },
    Audit {
        #[arg(long)]
        replay_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(long, value_enum, default_value_t = AuditCheck::Both)]
        check: AuditCheck,
        #[arg(long)]
        json_out: Option<PathBuf>,
        #[arg(long)]
        strict: bool,
        #[arg(long)]
        pretty: bool,
    },
    BuildBcDataset {
        #[arg(long)]
        replay_dir: PathBuf,
        #[arg(long)]
        maps_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        selected_scores: PathBuf,
        #[arg(long, default_value_t = 1)]
        workers: usize,
        #[arg(long)]
        top_selected: Option<usize>,
        #[arg(long, default_value_t = 32)]
        manifest_save_every: usize,
        #[arg(long, default_value_t = 16)]
        max_pending_writes: usize,
        #[arg(long, default_value_t = 16)]
        gc_every: usize,
        #[arg(long, default_value_t = 25)]
        status_every: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum AuditCheck {
    Raw,
    Dataset,
    Both,
}

#[derive(Debug, Clone, Serialize)]
struct AuditResult {
    path: String,
    raw_bsor_ok: bool,
    dataset_parse_ok: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ErrorCount {
    error: String,
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AuditSummary {
    count: usize,
    raw_bsor_ok: usize,
    dataset_parse_ok: usize,
    failure_count: usize,
    top_errors: Vec<ErrorCount>,
    failures: Vec<AuditResult>,
    replay_dir: String,
    check: AuditCheck,
}

fn emit_json<T: Serialize>(value: &T, pretty: bool) -> Result<()> {
    if pretty {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{}", serde_json::to_string(value)?);
    }
    Ok(())
}

fn read_stdin_text() -> Result<String> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    Ok(input)
}

fn load_replay_paths(dir: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let is_bsor = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("bsor"))
                .unwrap_or(false);
            if is_bsor {
                paths.push(path);
            }
        }
    }
    paths.sort();
    if limit > 0 && paths.len() > limit {
        paths.truncate(limit);
    }
    Ok(paths)
}

fn audit_one(path: &Path, check: AuditCheck) -> AuditResult {
    let mut result = AuditResult {
        path: path.display().to_string(),
        raw_bsor_ok: false,
        dataset_parse_ok: false,
        error: None,
    };

    match read_bsor_path(path) {
        Ok(replay) => {
            result.raw_bsor_ok = true;
            if matches!(check, AuditCheck::Dataset | AuditCheck::Both) {
                let view = dataset_view(&replay);
                let meta_ok = view.meta.song_hash.as_deref().unwrap_or("").trim().len() > 0;
                result.dataset_parse_ok = !view.frames.is_empty() && meta_ok;
                if !result.dataset_parse_ok {
                    result.error =
                        Some("dataset: dataset view returned no frames/meta".to_string());
                }
            }
        }
        Err(err) => {
            result.error = Some(format!("bsor:{}", err));
            if matches!(check, AuditCheck::Raw) {
                return result;
            }
        }
    }

    result
}

fn summarize(results: Vec<AuditResult>, replay_dir: &Path, check: AuditCheck) -> AuditSummary {
    let raw_bsor_ok = results.iter().filter(|item| item.raw_bsor_ok).count();
    let dataset_parse_ok = results.iter().filter(|item| item.dataset_parse_ok).count();
    let failures: Vec<_> = results
        .iter()
        .filter(|item| item.error.is_some())
        .cloned()
        .collect();

    let mut by_error: HashMap<String, usize> = HashMap::new();
    for failure in &failures {
        if let Some(error) = &failure.error {
            *by_error.entry(error.clone()).or_insert(0) += 1;
        }
    }
    let mut top_errors: Vec<_> = by_error
        .into_iter()
        .map(|(error, count)| ErrorCount { error, count })
        .collect();
    top_errors.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.error.cmp(&right.error))
    });
    top_errors.truncate(10);

    AuditSummary {
        count: results.len(),
        raw_bsor_ok,
        dataset_parse_ok,
        failure_count: failures.len(),
        top_errors,
        failures,
        replay_dir: replay_dir.display().to_string(),
        check,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::DumpJson { input, pretty } => {
            let replay = read_bsor_path(&input)?;
            emit_json(&replay, pretty)?;
        }
        Command::DumpDatasetJson { input, pretty } => {
            let replay = read_bsor_path(&input)?;
            let view = dataset_view(&replay);
            emit_json(&view, pretty)?;
        }
        Command::WriteJson { output, input_json } => {
            let payload = if let Some(input_path) = input_json {
                fs::read_to_string(&input_path)
                    .with_context(|| format!("failed to read {}", input_path.display()))?
            } else {
                read_stdin_text()?
            };
            let mut replay: bsor_tools::model::Bsor =
                serde_json::from_str(&payload).context("invalid BSOR JSON payload")?;
            for note in &mut replay.notes {
                note.refresh_derived_fields();
            }
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            write_bsor_path(&output, &replay)?;
        }
        Command::Validate { input, pretty } => {
            let replay = read_bsor_path(&input)?;
            let summary = validation_summary(&replay);
            emit_json(&summary, pretty)?;
        }
        Command::Audit {
            replay_dir,
            limit,
            check,
            json_out,
            strict,
            pretty,
        } => {
            let replay_files = load_replay_paths(&replay_dir, limit)?;
            if replay_files.is_empty() {
                anyhow::bail!("No .bsor files found in {}", replay_dir.display());
            }
            let results: Vec<_> = replay_files
                .iter()
                .map(|path| audit_one(path, check))
                .collect();
            let summary = summarize(results, &replay_dir, check);

            println!(
                "Scanned {} replays from {}",
                summary.count, summary.replay_dir
            );
            println!(
                "  Raw BSOR parse ok: {}/{}",
                summary.raw_bsor_ok, summary.count
            );
            if matches!(check, AuditCheck::Dataset | AuditCheck::Both) {
                println!(
                    "  Dataset parse ok:  {}/{}",
                    summary.dataset_parse_ok, summary.count
                );
            }
            println!("  Failures:          {}", summary.failure_count);
            if !summary.top_errors.is_empty() {
                println!("  Top errors:");
                for item in &summary.top_errors {
                    println!("    {:4}  {}", item.count, item.error);
                }
            }

            if let Some(json_path) = json_out {
                let text = if pretty {
                    serde_json::to_string_pretty(&summary)?
                } else {
                    serde_json::to_string(&summary)?
                };
                fs::write(&json_path, text)
                    .with_context(|| format!("failed to write {}", json_path.display()))?;
            }

            if strict && summary.failure_count > 0 {
                std::process::exit(1);
            }
        }
        Command::BuildBcDataset {
            replay_dir,
            maps_dir,
            output_dir,
            selected_scores,
            workers,
            top_selected,
            manifest_save_every,
            max_pending_writes,
            gc_every,
            status_every,
        } => {
            build_bc_dataset(BuildBcDatasetArgs {
                replay_dir,
                maps_dir,
                output_dir,
                selected_scores,
                workers,
                top_selected,
                manifest_save_every,
                max_pending_writes,
                gc_every,
                status_every,
            })?;
        }
    }

    Ok(())
}
