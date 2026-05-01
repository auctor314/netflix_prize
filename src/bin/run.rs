//! Pipeline runner. Reads a manifest like `pipeline-old.toml`, lists every
//! stage with its current status, and runs a single stage on demand. Status
//! is computed from input/output file existence; staleness checks and
//! transitive runs are intentionally out of scope for this initial version.

use indexmap::IndexMap;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, ExitCode};

const DEFAULT_PIPELINE: &str = "pipeline-old.toml";

#[derive(Deserialize, Debug, Default)]
struct Pipeline {
    #[serde(default)]
    split: HashMap<String, String>,
    #[serde(default)]
    defaults: HashMap<String, StageConfig>,
    #[serde(default)]
    stages: IndexMap<String, StageConfig>,
}

#[derive(Deserialize, Debug, Default, Clone)]
struct StageConfig {
    kind: Option<String>,
    #[serde(default)]
    inputs: Vec<String>,
    #[serde(default)]
    inputs_from: Vec<String>,
    #[serde(default)]
    outputs: Vec<String>,
    #[serde(default)]
    cmd: String,
    runner: Option<String>,
    config: Option<String>,
    target: Option<String>,
}

#[derive(Debug)]
struct ResolvedStage {
    inputs: Vec<String>,
    inputs_from: Vec<String>,
    outputs: Vec<String>,
    cmd: String,
}

#[derive(Debug)]
enum Status {
    Done,
    Ready,
    Blocked(Vec<String>),
}

fn merge_with_defaults(stage: &StageConfig, defaults: &HashMap<String, StageConfig>) -> StageConfig {
    let mut merged = stage.clone();
    if let Some(kind) = &stage.kind {
        if let Some(d) = defaults.get(kind) {
            if merged.inputs.is_empty() { merged.inputs = d.inputs.clone(); }
            if merged.inputs_from.is_empty() { merged.inputs_from = d.inputs_from.clone(); }
            if merged.outputs.is_empty() { merged.outputs = d.outputs.clone(); }
            if merged.cmd.is_empty() { merged.cmd = d.cmd.clone(); }
        }
    }
    merged
}

fn build_subst_vars(stage_name: &str, stage: &StageConfig, pipeline: &Pipeline) -> HashMap<String, String> {
    let mut vars: HashMap<String, String> = HashMap::new();
    for (k, v) in &pipeline.split {
        vars.insert(k.clone(), v.clone());
        vars.insert(format!("split.{}", k), v.clone());
    }
    vars.insert("name".to_string(), stage_name.to_string());
    if let Some(r) = &stage.runner { vars.insert("runner".to_string(), r.clone()); }
    if let Some(c) = &stage.config { vars.insert("config".to_string(), c.clone()); }
    if let Some(t) = &stage.target { vars.insert("target".to_string(), t.clone()); }
    vars
}

fn substitute(s: &str, vars: &HashMap<String, String>) -> String {
    let mut out = s.to_string();
    loop {
        let mut changed = false;
        for (k, v) in vars {
            let pat = format!("{{{}}}", k);
            if out.contains(&pat) {
                out = out.replace(&pat, v);
                changed = true;
            }
        }
        if !changed { break; }
    }
    out
}

fn resolve_pipeline(p: &Pipeline) -> IndexMap<String, ResolvedStage> {
    let mut out: IndexMap<String, ResolvedStage> = IndexMap::new();
    for (name, stage) in &p.stages {
        let merged = merge_with_defaults(stage, &p.defaults);
        let subst = build_subst_vars(name, &merged, p);
        out.insert(name.clone(), ResolvedStage {
            inputs: merged.inputs.iter().map(|s| substitute(s, &subst)).collect(),
            inputs_from: merged.inputs_from.clone(),
            outputs: merged.outputs.iter().map(|s| substitute(s, &subst)).collect(),
            cmd: substitute(&merged.cmd, &subst),
        });
    }
    out
}

fn path_exists(p: &str) -> bool {
    if p.ends_with('/') {
        Path::new(p).is_dir()
    } else {
        Path::new(p).exists()
    }
}

fn collect_all_inputs(stage: &ResolvedStage, resolved: &IndexMap<String, ResolvedStage>) -> Vec<String> {
    let mut all: Vec<String> = stage.inputs.clone();
    for up in &stage.inputs_from {
        if let Some(up_stage) = resolved.get(up) {
            all.extend(up_stage.outputs.iter().cloned());
        }
    }
    all
}

fn status_of(stage: &ResolvedStage, resolved: &IndexMap<String, ResolvedStage>) -> Status {
    let inputs = collect_all_inputs(stage, resolved);
    let missing: Vec<String> = inputs.iter().filter(|p| !path_exists(p)).cloned().collect();
    if !missing.is_empty() {
        return Status::Blocked(missing);
    }
    if stage.outputs.iter().all(|p| path_exists(p)) {
        Status::Done
    } else {
        Status::Ready
    }
}

fn list_stages(pipeline_path: &str, p: &Pipeline, resolved: &IndexMap<String, ResolvedStage>) {
    let split_name = p.split.get("name").cloned().unwrap_or_default();
    println!("Pipeline: {} (split = {})", pipeline_path, split_name);
    println!();
    for (name, stage) in resolved {
        let s = status_of(stage, resolved);
        let label = match &s {
            Status::Done => "DONE",
            Status::Ready => "READY",
            Status::Blocked(_) => "BLOCKED",
        };
        let detail = match &s {
            Status::Blocked(missing) => {
                let n = missing.len();
                if n == 1 {
                    format!("  (missing: {})", missing[0])
                } else {
                    format!("  (missing {}: {}, ...)", n, missing[0])
                }
            }
            _ => String::new(),
        };
        println!("  {:24} {}{}", name, label, detail);
    }
}

fn run_stage(stage_name: &str, resolved: &IndexMap<String, ResolvedStage>, force: bool) -> ExitCode {
    let stage = match resolved.get(stage_name) {
        Some(s) => s,
        None => {
            eprintln!("error: unknown stage '{}'", stage_name);
            let names: Vec<&str> = resolved.keys().map(|s| s.as_str()).collect();
            eprintln!("available: {}", names.join(", "));
            return ExitCode::from(2);
        }
    };
    match status_of(stage, resolved) {
        Status::Blocked(missing) => {
            eprintln!("error: stage '{}' is BLOCKED — missing inputs:", stage_name);
            for m in missing {
                eprintln!("  - {}", m);
            }
            return ExitCode::from(1);
        }
        Status::Done if !force => {
            println!("Stage '{}' is DONE — skipping. Use -f to force re-run.", stage_name);
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    println!("Running '{}': {}", stage_name, stage.cmd);
    match Command::new("sh").arg("-c").arg(&stage.cmd).status() {
        Ok(es) if es.success() => ExitCode::SUCCESS,
        Ok(es) => {
            eprintln!("stage exited with {}", es);
            ExitCode::from(es.code().unwrap_or(1) as u8)
        }
        Err(e) => {
            eprintln!("failed to spawn shell: {}", e);
            ExitCode::from(127)
        }
    }
}

fn print_help() {
    println!("Usage: run [-p FILE | -n] [-l] [-f] [STAGE]");
    println!();
    println!("  -p FILE, --pipeline FILE   pipeline manifest (default: {})", DEFAULT_PIPELINE);
    println!("  -n, --new                  shortcut for -p pipeline-new.toml");
    println!("  -l, --list                 list stages with status (default if no STAGE)");
    println!("  -f, --force                re-run STAGE even if its status is DONE");
    println!("  -h, --help                 show this help");
    println!("  STAGE                      run the named stage");
}

fn main() -> ExitCode {
    let mut pipeline_path = DEFAULT_PIPELINE.to_string();
    let mut stage_arg: Option<String> = None;
    let mut force_list = false;
    let mut force = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => { print_help(); return ExitCode::SUCCESS; }
            "-l" | "--list" => { force_list = true; i += 1; }
            "-f" | "--force" => { force = true; i += 1; }
            "-n" | "--new" => { pipeline_path = "pipeline-new.toml".to_string(); i += 1; }
            "-p" | "--pipeline" => {
                if i + 1 >= args.len() {
                    eprintln!("error: {} requires an argument", args[i]);
                    return ExitCode::from(2);
                }
                pipeline_path = args[i + 1].clone();
                i += 2;
            }
            s if s.starts_with('-') => {
                eprintln!("error: unknown flag '{}'", s);
                print_help();
                return ExitCode::from(2);
            }
            s => {
                if stage_arg.is_some() {
                    eprintln!("error: only one stage argument allowed");
                    return ExitCode::from(2);
                }
                stage_arg = Some(s.to_string());
                i += 1;
            }
        }
    }

    let content = match std::fs::read_to_string(&pipeline_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to read {}: {}", pipeline_path, e);
            return ExitCode::from(2);
        }
    };
    let pipeline: Pipeline = match toml::from_str(&content) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to parse {}: {}", pipeline_path, e);
            return ExitCode::from(2);
        }
    };
    let resolved = resolve_pipeline(&pipeline);

    match stage_arg {
        Some(name) => {
            if force_list {
                eprintln!("warning: -l ignored when a STAGE is specified");
            }
            run_stage(&name, &resolved, force)
        }
        None => {
            if force {
                eprintln!("warning: -f ignored without a STAGE");
            }
            list_stages(&pipeline_path, &pipeline, &resolved);
            ExitCode::SUCCESS
        }
    }
}
