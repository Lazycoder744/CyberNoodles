use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::cast_slice;
use half::f16;
use md5::compute as md5_compute;
use rayon::prelude::*;
use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use zip::ZipArchive;

use crate::io::read_bsor_path;
use crate::sanitize::{dataset_view, DatasetFrame};

const MANIFEST_VERSION: u32 = 16;
const MANIFEST_SEMANTIC_SCHEMA_ID: &str = "bc-shard-semantics-v1";
const NOTE_LOOKAHEAD_BEATS: f32 = 1.25;
const FOLLOWTHROUGH_BEATS: f32 = 0.35;
const BACKGROUND_FRAME_STRIDE: usize = 6;
const VAL_FRACTION: f32 = 0.10;
const MIN_REPLAY_FRAMES: usize = 16;
const MIN_SAMPLE_FRAMES: usize = 8;
const TARGET_POSE_HORIZON_FRAMES: usize = 2;
const SIM_SAMPLE_HZ: f32 = 60.0;
const SIM_SAMPLE_DT: f32 = 1.0 / SIM_SAMPLE_HZ;
const SIM_NOTE_TIME_MIN_BEATS: f32 = -1.0;
const SIM_NOTE_TIME_MAX_BEATS: f32 = 4.0;
const NOTE_TIME_FEATURE_MIN_BEATS: f32 = -1.0;
const NOTE_TIME_FEATURE_MAX_BEATS: f32 = 8.0;
const SIM_OBSTACLE_TIME_MIN_BEATS: f32 = -1.0;
const SIM_OBSTACLE_TIME_MAX_BEATS: f32 = 6.0;
const NOTE_FEATURE_LAYOUT: &str =
    "spawn_visible_contact_shifted_beat_time+physical_time+physical_z";
const TRACK_Z_BASE: f32 = 0.9;
const FLOAT16_MAX: f32 = 65504.0;
const INITIAL_HALF_JUMP_BEATS: f32 = 4.0;
const MIN_HALF_JUMP_BEATS: f32 = 1.0;
const MAX_HALF_JUMP_DISTANCE_METERS: f32 = 18.0;

const INPUT_DIM: usize = 362;
const NOTE_FEATURES: usize = 10;
const NUM_UPCOMING_NOTES: usize = 20;
const NOTES_DIM: usize = NOTE_FEATURES * NUM_UPCOMING_NOTES;
const OBSTACLE_FEATURES: usize = 6;
const NUM_UPCOMING_OBSTACLES: usize = 6;
const OBSTACLES_DIM: usize = OBSTACLE_FEATURES * NUM_UPCOMING_OBSTACLES;
const POSE_DIM: usize = 21;
const VELOCITY_DIM: usize = 21;
const STATE_HISTORY_OFFSETS: [usize; 3] = [0, 2, 4];

const SCORE_CLASS_NORMAL: f32 = 0.0;
const SCORE_CLASS_ARC_HEAD: f32 = 1.0;
const SCORE_CLASS_ARC_TAIL: f32 = 2.0;
const SCORE_CLASS_CHAIN_HEAD: f32 = 3.0;
const SCORE_CLASS_CHAIN_LINK: f32 = 4.0;
const ACTION_ABS_COMPONENT_LIMIT: f32 = 2.0;
const STATE_POSE_POSITION_ABS_LIMIT: f32 = ACTION_ABS_COMPONENT_LIMIT;
const FEATURE_STORAGE_DTYPE: &str = "float16";
const TARGET_STORAGE_DTYPE: &str = "float32";
const SHARD_STORAGE_DTYPE: &str = "features=float16,targets=float32";
const TARGET_ACTION_DELTA_CLAMP: [f32; POSE_DIM] = [
    0.08, 0.08, 0.08, 0.045, 0.045, 0.045, 0.045, 0.12, 0.12, 0.12, 0.07, 0.07, 0.07, 0.07, 0.12,
    0.12, 0.12, 0.07, 0.07, 0.07, 0.07,
];

#[derive(Debug, Clone)]
pub struct BuildBcDatasetArgs {
    pub replay_dir: PathBuf,
    pub maps_dir: PathBuf,
    pub output_dir: PathBuf,
    pub selected_scores: PathBuf,
    pub workers: usize,
    pub top_selected: Option<usize>,
    pub manifest_save_every: usize,
    pub max_pending_writes: usize,
    pub gc_every: usize,
    pub status_every: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ManifestCounts {
    #[serde(default)]
    train_samples: usize,
    #[serde(default)]
    val_samples: usize,
    #[serde(default)]
    train_replays: usize,
    #[serde(default)]
    val_replays: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestScoreClassValues {
    normal: f32,
    arc_head: f32,
    arc_tail: f32,
    chain_head: f32,
    chain_link: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestActionContract {
    action_dim: usize,
    action_representation: String,
    policy_mean_contract: String,
    simulator_consumption: String,
    absolute_component_limit: f32,
    per_step_delta_clamp: Vec<f32>,
    quaternion_slices: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestTargetContract {
    target_representation: String,
    target_generation: String,
    future_pose_horizon_frames: usize,
    normalizes_quaternions: bool,
    stored_dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestMissingNoteSentinel {
    time: f32,
    line_index: f32,
    line_layer: f32,
    note_type: f32,
    cut_dx: f32,
    cut_dy: f32,
    score_class: f32,
    score_cap: f32,
    time_seconds: f32,
    z_distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestSentinelContract {
    missing_note: ManifestMissingNoteSentinel,
    hidden_future_note: String,
    missing_obstacle: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestFollowthroughContract {
    keep_next_note_window_beats: f32,
    keep_previous_note_window_beats: f32,
    keep_background_frame_stride: usize,
    timing_reference: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestSemanticSchema {
    schema_id: String,
    sample_hz: f32,
    state_history_offsets: Vec<usize>,
    note_feature_layout: String,
    note_lookahead_beats: f32,
    followthrough_beats: f32,
    background_frame_stride: usize,
    target_pose_horizon_frames: usize,
    sim_note_time_range_beats: Vec<f32>,
    #[serde(default)]
    note_time_feature_range_beats: Vec<f32>,
    sim_obstacle_time_range_beats: Vec<f32>,
    num_upcoming_notes: usize,
    note_features: usize,
    num_upcoming_obstacles: usize,
    obstacle_features: usize,
    pose_dim: usize,
    velocity_dim: usize,
    track_z_base: f32,
    shard_storage_dtype: String,
    #[serde(default)]
    feature_storage_dtype: String,
    action_contract: ManifestActionContract,
    target_contract: ManifestTargetContract,
    sentinel_contract: ManifestSentinelContract,
    followthrough_contract: ManifestFollowthroughContract,
    score_class_values: ManifestScoreClassValues,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestProvenanceSchema {
    source: String,
    shard_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ManifestWarning {
    code: String,
    message: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    replay_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ManifestShardRecord {
    split: String,
    song_hash: String,
    #[serde(default)]
    difficulty: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    bpm: Option<f32>,
    #[serde(default)]
    njs: Option<f32>,
    #[serde(default)]
    jump_offset: Option<f32>,
    replay_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    x_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    y_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shard_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tensor_format: Option<String>,
    samples: usize,
    #[serde(default)]
    frames_total: usize,
    #[serde(default)]
    frames_total_raw: usize,
    #[serde(default)]
    frames_total_resampled: usize,
    #[serde(default)]
    sample_hz: f32,
    #[serde(default)]
    samples_dropped: usize,
    #[serde(default)]
    note_window_samples: usize,
    #[serde(default)]
    background_samples: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    player_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    player_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_timestamp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    game_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    platform: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tracking_system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hmd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    controller: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    left_handed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    player_height: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_start_time: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_fail_time: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    feature_dim: usize,
    target_dim: usize,
    target_pose_horizon_frames: usize,
    #[serde(default)]
    semantic_schema: Option<ManifestSemanticSchema>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provenance_schema: Option<ManifestProvenanceSchema>,
    sample_hz: f32,
    history_offsets: Vec<usize>,
    note_feature_layout: String,
    note_lookahead_beats: f32,
    followthrough_beats: f32,
    background_stride: usize,
    #[serde(default)]
    shard_format: Option<String>,
    #[serde(default)]
    done: Vec<String>,
    #[serde(default)]
    failed: Vec<String>,
    #[serde(default)]
    warnings: Vec<ManifestWarning>,
    #[serde(default)]
    shards: Vec<ManifestShardRecord>,
    #[serde(default)]
    counts: ManifestCounts,
}

#[derive(Debug, Clone)]
struct BeatmapNote {
    index: i32,
    time: f32,
    line_index: f32,
    line_layer: f32,
    note_type: i32,
    cut_direction: i32,
    arc_head: bool,
    arc_tail: bool,
    chain_head: bool,
    chain_link: bool,
    score_class: f32,
    score_cap: f32,
}

#[derive(Debug, Clone)]
struct BeatmapObstacle {
    time: f32,
    line_index: f32,
    line_layer: f32,
    width: f32,
    height: f32,
    duration: f32,
}

#[derive(Debug, Clone)]
struct ArcDef {
    color: i32,
    head_time: f32,
    head_line_index: f32,
    head_line_layer: f32,
    tail_time: f32,
    tail_line_index: f32,
    tail_line_layer: f32,
}

#[derive(Debug, Clone)]
struct ChainDef {
    color: i32,
    head_time: f32,
    head_line_index: f32,
    head_line_layer: f32,
    tail_time: f32,
    tail_line_index: f32,
    tail_line_layer: f32,
    slice_count: usize,
    squish: f32,
}

#[derive(Debug, Clone)]
struct BeatmapData {
    notes: Vec<BeatmapNote>,
    obstacles: Vec<BeatmapObstacle>,
    mode: String,
    difficulty: Option<String>,
    njs: f32,
    offset: f32,
    song_name: String,
    song_author_name: String,
    level_author_name: String,
    environment_name: String,
}

#[derive(Debug, Clone)]
struct DifficultyEntry {
    mode: String,
    difficulty: String,
    filename: String,
    rank: i32,
    note_jump_movement_speed: Option<f32>,
    note_jump_start_beat_offset: Option<f32>,
}

#[derive(Debug, Clone)]
struct ResolvedBeatmap {
    beatmap: Arc<BeatmapData>,
    bpm: f32,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct MapRequestKey {
    map_hash: String,
    preferred_difficulty: String,
    preferred_mode: String,
    strict_mode: bool,
}

#[derive(Debug)]
struct MapCache {
    maps_dir: PathBuf,
    resolved: Mutex<HashMap<MapRequestKey, Option<Arc<ResolvedBeatmap>>>>,
}

#[derive(Debug, Clone)]
struct PoseFrame {
    time: f32,
    pose: [f32; POSE_DIM],
}

#[derive(Debug, Clone)]
struct ExtractedFeatures {
    features: Vec<f32>,
    targets: Vec<f32>,
    sample_count: usize,
    frames_total: usize,
    samples_dropped: usize,
    note_window_samples: usize,
    background_samples: usize,
}

#[derive(Debug, Clone)]
struct ReplayManifestMeta {
    song_hash: String,
    difficulty: Option<String>,
    mode: String,
    bpm: f32,
    njs: f32,
    jump_offset: f32,
    frames_total: usize,
    frames_total_raw: usize,
    frames_total_resampled: usize,
    sample_hz: f32,
    samples_dropped: usize,
    note_window_samples: usize,
    background_samples: usize,
    player_id: Option<String>,
    player_name: Option<String>,
    replay_timestamp: Option<String>,
    game_version: Option<String>,
    platform: Option<String>,
    tracking_system: Option<String>,
    hmd: Option<String>,
    controller: Option<String>,
    score: Option<u32>,
    left_handed: Option<bool>,
    player_height: Option<f32>,
    replay_start_time: Option<f32>,
    replay_fail_time: Option<f32>,
}

#[derive(Debug, Clone)]
struct ReplaySuccess {
    name: String,
    split: String,
    replay_meta: ReplayManifestMeta,
    shard_path: String,
    sample_count: usize,
}

#[derive(Debug, Clone)]
struct ReplayFailure {
    name: String,
    reason: String,
}

#[derive(Debug, Clone)]
enum ReplayResult {
    Success(ReplaySuccess),
    Failed(ReplayFailure),
}

struct ProgressState {
    total: usize,
    status_every: usize,
    started_at: Instant,
    completed: AtomicUsize,
    succeeded: AtomicUsize,
    failed: AtomicUsize,
    print_lock: Mutex<()>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct NoteKey {
    time_micros: i64,
    line_index_e4: i64,
    line_layer_e4: i64,
    note_type: i32,
}

impl MapCache {
    fn new(maps_dir: PathBuf) -> Self {
        Self {
            maps_dir,
            resolved: Mutex::new(HashMap::new()),
        }
    }

    fn get_map_data(
        &self,
        map_hash: &str,
        preferred_difficulty: Option<&str>,
        preferred_mode: Option<&str>,
    ) -> Result<Option<Arc<ResolvedBeatmap>>> {
        let key = MapRequestKey {
            map_hash: map_hash.to_ascii_uppercase(),
            preferred_difficulty: normalize_difficulty_name(preferred_difficulty.unwrap_or("")),
            preferred_mode: normalize_mode_name(preferred_mode.unwrap_or("Standard")),
            strict_mode: preferred_mode.is_some(),
        };

        if let Some(cached) = self.resolved.lock().unwrap().get(&key).cloned() {
            return Ok(cached);
        }

        let loaded = self.load_map(map_hash, preferred_difficulty, preferred_mode)?;
        self.resolved.lock().unwrap().insert(key, loaded.clone());
        Ok(loaded)
    }

    fn load_map(
        &self,
        map_hash: &str,
        preferred_difficulty: Option<&str>,
        preferred_mode: Option<&str>,
    ) -> Result<Option<Arc<ResolvedBeatmap>>> {
        let Some(zip_path) = resolve_map_zip_path(&self.maps_dir, map_hash) else {
            return Ok(None);
        };

        let file = File::open(&zip_path)
            .with_context(|| format!("failed to open map archive {}", zip_path.display()))?;
        let mut archive = ZipArchive::new(file)
            .with_context(|| format!("failed to read zip archive {}", zip_path.display()))?;

        let Some(info_name) = find_info_file_name(&mut archive)? else {
            return Ok(None);
        };
        let info_bytes = read_zip_entry_bytes(&mut archive, &info_name)?;
        let info_data: Value = serde_json::from_slice(&info_bytes)
            .with_context(|| format!("failed to parse Info.dat in {}", zip_path.display()))?;
        let bpm = get_f32_any(&info_data, &["_beatsPerMinute", "beatsPerMinute"]).unwrap_or(120.0);
        let entries = collect_difficulty_entries(&info_data);
        let selected_entry = select_difficulty_entry(
            &entries,
            preferred_mode,
            preferred_difficulty,
            preferred_mode.is_some(),
        );

        let dat_file = if let Some(entry) = selected_entry.as_ref() {
            Some(entry.filename.clone())
        } else if preferred_mode.is_some() {
            None
        } else {
            find_first_dat_file(&mut archive)?
        };

        let Some(dat_file) = dat_file else {
            return Ok(None);
        };

        let dat_name = resolve_zip_entry_name(&mut archive, &dat_file)
            .ok_or_else(|| anyhow!("missing beatmap dat {} in {}", dat_file, zip_path.display()))?;
        let dat_bytes = read_zip_entry_bytes(&mut archive, &dat_name)?;
        let mut beatmap = parse_beatmap_dat(&dat_bytes)?;

        let fallback_njs = get_f32_any(
            &info_data,
            &["_noteJumpMovementSpeed", "noteJumpMovementSpeed"],
        )
        .unwrap_or(18.0);
        let fallback_offset = get_f32_any(
            &info_data,
            &["_noteJumpStartBeatOffset", "noteJumpStartBeatOffset"],
        )
        .unwrap_or(0.0);
        let selected_mode = selected_entry
            .as_ref()
            .map(|entry| entry.mode.clone())
            .unwrap_or_else(|| preferred_mode.unwrap_or("Standard").to_string());
        let selected_difficulty = selected_entry
            .as_ref()
            .map(|entry| entry.difficulty.clone())
            .or_else(|| preferred_difficulty.map(ToOwned::to_owned));
        beatmap.mode = selected_mode;
        beatmap.difficulty = selected_difficulty;
        beatmap.njs = selected_entry
            .as_ref()
            .and_then(|entry| entry.note_jump_movement_speed)
            .unwrap_or(fallback_njs);
        beatmap.offset = selected_entry
            .as_ref()
            .and_then(|entry| entry.note_jump_start_beat_offset)
            .unwrap_or(fallback_offset);
        beatmap.song_name = get_string_any(
            &info_data,
            &["_songName", "songName", "_songSubName", "songSubName"],
        )
        .unwrap_or_else(|| map_hash.to_string());
        beatmap.song_author_name =
            get_string_any(&info_data, &["_songAuthorName", "songAuthorName"]).unwrap_or_default();
        beatmap.level_author_name =
            get_string_any(&info_data, &["_levelAuthorName", "levelAuthorName"])
                .unwrap_or_default();
        beatmap.environment_name =
            get_string_any(&info_data, &["_environmentName", "environmentName"])
                .unwrap_or_else(|| "DefaultEnvironment".to_string());

        Ok(Some(Arc::new(ResolvedBeatmap {
            beatmap: Arc::new(beatmap),
            bpm,
        })))
    }
}

impl ProgressState {
    fn new(total: usize, status_every: usize) -> Self {
        Self {
            total,
            status_every: status_every.max(1),
            started_at: Instant::now(),
            completed: AtomicUsize::new(0),
            succeeded: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            print_lock: Mutex::new(()),
        }
    }

    fn note_result(&self, success: bool) {
        if success {
            self.succeeded.fetch_add(1, AtomicOrdering::Relaxed);
        } else {
            self.failed.fetch_add(1, AtomicOrdering::Relaxed);
        }
        let completed = self.completed.fetch_add(1, AtomicOrdering::SeqCst) + 1;
        if completed == self.total || (completed % self.status_every) == 0 {
            let _guard = self.print_lock.lock().unwrap();
            let succeeded = self.succeeded.load(AtomicOrdering::Relaxed);
            let failed = self.failed.load(AtomicOrdering::Relaxed);
            let elapsed = self.started_at.elapsed().as_secs_f64().max(1e-6);
            let rate_per_min = (completed as f64 / elapsed) * 60.0;
            let remaining = self.total.saturating_sub(completed);
            let eta_minutes = if rate_per_min > 0.0 {
                remaining as f64 / rate_per_min
            } else {
                f64::INFINITY
            };
            let eta_text = if eta_minutes.is_finite() {
                format!("{:.1}h", eta_minutes / 60.0)
            } else {
                "unknown".to_string()
            };
            println!(
                "Progress {completed}/{} | ok {succeeded} | failed {failed} | {:.2} replays/min | ETA {eta_text}",
                self.total,
                rate_per_min
            );
        }
    }
}

pub fn build_bc_dataset(args: BuildBcDatasetArgs) -> Result<()> {
    let workers = args.workers.max(1);
    let status_every = args.status_every.max(1);
    let manifest_save_every = args.manifest_save_every.max(1);
    let _max_pending_writes = args.max_pending_writes.max(1);
    let _gc_every = args.gc_every;

    let shard_root = args.output_dir.join("bc_shards");
    fs::create_dir_all(shard_root.join("train"))?;
    fs::create_dir_all(shard_root.join("val"))?;

    let mut replay_files = load_replay_paths(&args.replay_dir)?;
    let selected_subset = load_selected_replay_subset(&args.selected_scores, args.top_selected)?;
    let missing_selected: Vec<String> = selected_subset
        .as_ref()
        .map(|selected| {
            let local_names: HashSet<String> = replay_files
                .iter()
                .filter_map(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.to_string())
                })
                .collect();
            selected
                .iter()
                .filter(|name| !local_names.contains(*name))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    if let Some(selected) = selected_subset.as_ref() {
        let selected_names: HashSet<_> = selected.iter().cloned().collect();
        replay_files.retain(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| selected_names.contains(name))
                .unwrap_or(false)
        });
    }
    let total_replays = replay_files.len();

    let manifest_path = shard_root.join("manifest.json");
    let mut manifest = load_manifest(&manifest_path)?;
    let mut manifest_dirty = false;
    if selected_subset.is_some() && !missing_selected.is_empty() {
        let message = format!(
            "{} selected replay(s) are missing locally and will not be built.",
            missing_selected.len()
        );
        println!(
            "Warning: {message} {}",
            replay_preview(&missing_selected, 5)
        );
        manifest_dirty |= set_manifest_warning(
            &mut manifest,
            ManifestWarning {
                code: "selected_replays_missing".to_string(),
                message,
                replay_files: missing_selected.clone(),
                count: Some(missing_selected.len()),
            },
        );
    } else if selected_subset.is_some() {
        manifest_dirty |= clear_manifest_warning(&mut manifest, "selected_replays_missing");
    }
    if manifest_dirty {
        save_manifest(&manifest, &manifest_path)?;
    }
    let done_set: HashSet<_> = manifest.done.iter().cloned().collect();
    let failed_history_count = manifest.failed.len();
    let remaining = pending_replay_paths(replay_files, &done_set);

    println!(
        "Found {} replays total. {} already processed, {} failed history, {} remaining.",
        total_replays,
        done_set.len(),
        failed_history_count,
        remaining.len()
    );
    if let Some(selected) = selected_subset.as_ref() {
        println!(
            "Subset mode: top {} replay(s) from {}.",
            selected.len(),
            args.selected_scores.display()
        );
    }
    if remaining.is_empty() {
        print_final_summary(&manifest);
        return Ok(());
    }

    println!("Using {workers} worker thread(s) for native Rust dataset building.");
    let map_cache = Arc::new(MapCache::new(args.maps_dir.clone()));
    let progress = Arc::new(ProgressState::new(remaining.len(), status_every));
    let shard_root_arc = Arc::new(shard_root);
    let manifest_state = Arc::new(Mutex::new((manifest, 0_usize)));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("failed to create rayon thread pool")?;

    pool.install(|| {
        remaining.par_iter().try_for_each(|path| -> Result<()> {
            let result = process_and_save_replay(path, &shard_root_arc, &map_cache);
            progress.note_result(matches!(result, ReplayResult::Success(_)));
            let mut state = manifest_state.lock().unwrap();
            let (manifest, processed_since_save) = &mut *state;
            match result {
                ReplayResult::Success(payload) => record_manifest_success(manifest, payload),
                ReplayResult::Failed(payload) => {
                    if !manifest.failed.iter().any(|name| name == &payload.name) {
                        manifest.failed.push(payload.name.clone());
                    }
                    println!("  Skipping {}: {}", payload.name, payload.reason);
                }
            }
            *processed_since_save += 1;
            if *processed_since_save >= manifest_save_every {
                save_manifest(manifest, &manifest_path)?;
                *processed_since_save = 0;
            }
            Ok(())
        })
    })?;

    let mut state = manifest_state.lock().unwrap();
    save_manifest(&state.0, &manifest_path)?;
    state.1 = 0;
    print_final_summary(&state.0);
    Ok(())
}

fn print_final_summary(manifest: &Manifest) {
    let counts = &manifest.counts;
    println!("\nDataset build complete.");
    println!(
        "  Train: {} replays, {} samples",
        counts.train_replays,
        format_count(counts.train_samples)
    );
    println!(
        "  Val:   {} replays, {} samples",
        counts.val_replays,
        format_count(counts.val_samples)
    );
    println!("  Shards: {}", manifest.shards.len());
}

fn process_and_save_replay(
    replay_path: &Path,
    shard_root: &Path,
    map_cache: &MapCache,
) -> ReplayResult {
    let name = replay_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("unknown.bsor")
        .to_string();

    match process_single(replay_path, map_cache).and_then(|(replay_meta, features)| {
        let split = assign_split(&replay_meta.song_hash).to_string();
        let shard_path = write_replay_shard_file(
            &split,
            &name,
            &features.features,
            &features.targets,
            features.sample_count,
            shard_root,
        )?;
        Ok(ReplaySuccess {
            name: name.clone(),
            split,
            replay_meta: ReplayManifestMeta {
                frames_total: features.frames_total,
                samples_dropped: features.samples_dropped,
                note_window_samples: features.note_window_samples,
                background_samples: features.background_samples,
                ..replay_meta
            },
            shard_path,
            sample_count: features.sample_count,
        })
    }) {
        Ok(success) => ReplayResult::Success(success),
        Err(err) => ReplayResult::Failed(ReplayFailure {
            name,
            reason: format!("{err:#}"),
        }),
    }
}

fn process_single(
    replay_path: &Path,
    map_cache: &MapCache,
) -> Result<(ReplayManifestMeta, ExtractedFeatures)> {
    let replay = read_bsor_path(replay_path)
        .with_context(|| format!("failed to parse BSOR {}", replay_path.display()))?;
    let dataset = dataset_view(&replay);
    let song_hash = dataset
        .meta
        .song_hash
        .clone()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("missing song hash"))?;

    let replay_difficulty = dataset
        .meta
        .difficulty
        .clone()
        .filter(|value| !value.trim().is_empty());
    let replay_mode = if dataset.meta.mode.trim().is_empty() {
        "Standard".to_string()
    } else {
        dataset.meta.mode.clone()
    };

    if normalize_mode_name(&replay_mode) != "standard" {
        bail!("non-standard replay");
    }
    if !dataset.meta.modifiers.trim().is_empty() {
        bail!("modified replay");
    }

    let frames_total_raw = dataset.frames.len();
    let frames = convert_dataset_frames(&dataset.frames)?;
    let frames = resample_frames_to_sim_rate(&frames, SIM_SAMPLE_DT);
    let frames_total_resampled = frames.len();

    let resolved = map_cache
        .get_map_data(&song_hash, replay_difficulty.as_deref(), Some(&replay_mode))?
        .ok_or_else(|| anyhow!("missing map/difficulty for {}", song_hash))?;

    let extracted = extract_features(&frames, &resolved.beatmap, resolved.bpm)?;
    if extracted.sample_count < MIN_SAMPLE_FRAMES {
        bail!("too few usable samples");
    }

    Ok((
        ReplayManifestMeta {
            song_hash: song_hash.to_ascii_uppercase(),
            difficulty: resolved.beatmap.difficulty.clone(),
            mode: resolved.beatmap.mode.clone(),
            bpm: resolved.bpm,
            njs: resolved.beatmap.njs,
            jump_offset: resolved.beatmap.offset,
            frames_total: extracted.frames_total,
            frames_total_raw,
            frames_total_resampled,
            sample_hz: SIM_SAMPLE_HZ,
            samples_dropped: extracted.samples_dropped,
            note_window_samples: extracted.note_window_samples,
            background_samples: extracted.background_samples,
            player_id: optional_non_empty(&replay.info.player_id),
            player_name: optional_non_empty(&replay.info.player_name),
            replay_timestamp: optional_non_empty(&replay.info.timestamp),
            game_version: optional_non_empty(&replay.info.game_version),
            platform: optional_non_empty(&replay.info.platform),
            tracking_system: optional_non_empty(&replay.info.tracking_system),
            hmd: optional_non_empty(&replay.info.hmd),
            controller: optional_non_empty(&replay.info.controller),
            score: Some(replay.info.score),
            left_handed: Some(replay.info.left_handed),
            player_height: finite_f32_option(replay.info.height),
            replay_start_time: finite_f32_option(replay.info.start_time),
            replay_fail_time: finite_f32_option(replay.info.fail_time),
        },
        extracted,
    ))
}

fn optional_non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn finite_f32_option(value: f32) -> Option<f32> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn apply_state_pose_contract(pose: &mut [f32; POSE_DIM]) {
    for &(start, end) in &[(0_usize, 3_usize), (7, 10), (14, 17)] {
        for value in pose[start..end].iter_mut() {
            *value = value.clamp(-STATE_POSE_POSITION_ABS_LIMIT, STATE_POSE_POSITION_ABS_LIMIT);
        }
    }
    normalize_pose_quaternions(pose);
}

fn convert_dataset_frames(frames: &[DatasetFrame]) -> Result<Vec<PoseFrame>> {
    let mut output = Vec::with_capacity(frames.len());
    for frame in frames {
        if frame.pose.len() < POSE_DIM {
            bail!("dataset view returned pose shorter than {POSE_DIM}");
        }
        let mut pose = [0.0_f32; POSE_DIM];
        pose.copy_from_slice(&frame.pose[..POSE_DIM]);
        apply_state_pose_contract(&mut pose);
        output.push(PoseFrame {
            time: frame.time,
            pose,
        });
    }
    Ok(output)
}

fn extract_features(
    frames: &[PoseFrame],
    beatmap: &BeatmapData,
    bpm: f32,
) -> Result<ExtractedFeatures> {
    if frames.len() < (MIN_REPLAY_FRAMES + TARGET_POSE_HORIZON_FRAMES) || beatmap.notes.len() < 2 {
        return Ok(ExtractedFeatures {
            features: Vec::new(),
            targets: Vec::new(),
            sample_count: 0,
            frames_total: frames.len(),
            samples_dropped: 0,
            note_window_samples: 0,
            background_samples: 0,
        });
    }

    let bps_f32 = (bpm / 60.0).max(1e-6);
    let bps = (f64::from(bpm) / 60.0).max(1e-6);
    let note_jump_speed = f64::from(if beatmap.njs == 0.0 {
        18.0
    } else {
        beatmap.njs
    });
    let spawn_ahead_beats = f64::from(compute_spawn_ahead_beats(
        bpm,
        if beatmap.njs == 0.0 {
            18.0
        } else {
            beatmap.njs
        },
        beatmap.offset,
    ));
    let note_times: Vec<f32> = beatmap.notes.iter().map(|note| note.time).collect();
    let obstacle_times: Vec<f32> = beatmap
        .obstacles
        .iter()
        .map(|obstacle| obstacle.time)
        .collect();

    let total_candidates = frames.len().saturating_sub(TARGET_POSE_HORIZON_FRAMES);
    let mut features = Vec::with_capacity(total_candidates * INPUT_DIM);
    let mut targets = Vec::with_capacity(total_candidates * POSE_DIM);
    let mut note_idx = 0_usize;
    let mut obstacle_idx = 0_usize;
    let mut kept_note_window = 0_usize;
    let mut kept_background = 0_usize;

    for current_idx in 0..total_candidates {
        let t_beat = f64::from(frames[current_idx].time * bps_f32);
        let contact_shift_beats = ((f64::from(TRACK_Z_BASE)
            + f64::from(frames[current_idx].pose[2]))
            / note_jump_speed.max(1e-6))
            * bps;

        while note_idx < beatmap.notes.len()
            && (f64::from(note_times[note_idx])
                + contact_shift_beats
                + f64::from(FOLLOWTHROUGH_BEATS))
                < t_beat
        {
            note_idx += 1;
        }
        while obstacle_idx < beatmap.obstacles.len()
            && f64::from(obstacle_times[obstacle_idx] + beatmap.obstacles[obstacle_idx].duration)
                < t_beat
        {
            obstacle_idx += 1;
        }

        let next_delta = if note_idx < note_times.len() {
            (f64::from(note_times[note_idx]) + contact_shift_beats) - t_beat
        } else {
            99.0
        };
        let prev_delta = if note_idx > 0 {
            t_beat - (f64::from(note_times[note_idx - 1]) + contact_shift_beats)
        } else {
            99.0
        };
        if !should_keep_frame(current_idx, next_delta as f32, prev_delta as f32) {
            continue;
        }

        let mut sample = [0.0_f32; INPUT_DIM];
        build_note_feature_vector(
            &mut sample[..NOTES_DIM],
            &beatmap.notes,
            note_idx,
            t_beat,
            bps,
            note_jump_speed,
            frames[current_idx].pose[2],
            spawn_ahead_beats,
        );
        build_obstacle_feature_vector(
            &mut sample[NOTES_DIM..NOTES_DIM + OBSTACLES_DIM],
            &beatmap.obstacles,
            obstacle_idx,
            t_beat,
            spawn_ahead_beats,
        );

        let mut cursor = NOTES_DIM + OBSTACLES_DIM;
        for offset in STATE_HISTORY_OFFSETS {
            let hist_idx = current_idx.saturating_sub(offset);
            sample[cursor..cursor + POSE_DIM].copy_from_slice(&frames[hist_idx].pose);
            cursor += POSE_DIM;

            let velocity = frame_velocity(frames, hist_idx);
            sample[cursor..cursor + VELOCITY_DIM].copy_from_slice(&velocity);
            cursor += VELOCITY_DIM;
        }
        if cursor != INPUT_DIM {
            bail!("feature width mismatch: expected {INPUT_DIM}, got {cursor}");
        }

        let target_idx = current_idx + TARGET_POSE_HORIZON_FRAMES;
        let target_pose =
            simulator_executable_pose_target(&frames[current_idx].pose, &frames[target_idx].pose);
        if !fits_float16_slice(&sample) || !fits_float16_slice(&target_pose) {
            continue;
        }

        features.extend_from_slice(&sample);
        targets.extend_from_slice(&target_pose);
        if (next_delta as f32) <= NOTE_LOOKAHEAD_BEATS || (prev_delta as f32) <= FOLLOWTHROUGH_BEATS
        {
            kept_note_window += 1;
        } else {
            kept_background += 1;
        }
    }

    let sample_count = features.len() / INPUT_DIM;
    Ok(ExtractedFeatures {
        sample_count,
        features,
        targets,
        frames_total: frames.len(),
        samples_dropped: total_candidates.saturating_sub(sample_count),
        note_window_samples: kept_note_window,
        background_samples: kept_background,
    })
}

fn build_note_feature_vector(
    out: &mut [f32],
    notes: &[BeatmapNote],
    note_idx: usize,
    t_beat: f64,
    bps: f64,
    note_jump_speed: f64,
    head_z: f32,
    spawn_ahead_beats: f64,
) {
    for offset in 0..NUM_UPCOMING_NOTES {
        let base = offset * NOTE_FEATURES;
        if note_idx + offset < notes.len() {
            let note = &notes[note_idx + offset];
            let raw_time_offset = f64::from(note.time) - t_beat;
            if raw_time_offset > spawn_ahead_beats.max(0.0) {
                out[base] = 0.0;
                out[base + 1] = 0.0;
                out[base + 2] = 0.0;
                out[base + 3] = -1.0;
                out[base + 4] = 0.0;
                out[base + 5] = 0.0;
                out[base + 6] = 0.0;
                out[base + 7] = 0.0;
                out[base + 8] = 0.0;
                out[base + 9] = 0.0;
                continue;
            }
            let time_offset = clamp_f64(
                raw_time_offset,
                f64::from(SIM_NOTE_TIME_MIN_BEATS),
                f64::from(SIM_NOTE_TIME_MAX_BEATS),
            );
            let safe_njs = note_jump_speed.max(1e-6);
            let safe_head_z = f64::from(head_z);
            let mut time_seconds = (time_offset / bps.max(1e-6))
                + ((f64::from(TRACK_Z_BASE) + safe_head_z) / safe_njs);
            let contact_time_beats = clamp_f64(
                time_seconds * bps,
                f64::from(NOTE_TIME_FEATURE_MIN_BEATS),
                f64::from(NOTE_TIME_FEATURE_MAX_BEATS),
            );
            time_seconds = contact_time_beats / bps.max(1e-6);
            let z_distance = time_seconds * safe_njs;
            let (dx, dy) = encode_cut_direction(note.cut_direction);
            out[base] = contact_time_beats as f32;
            out[base + 1] = note.line_index;
            out[base + 2] = note.line_layer;
            out[base + 3] = note.note_type as f32;
            out[base + 4] = dx;
            out[base + 5] = dy;
            out[base + 6] = note.score_class;
            out[base + 7] = note.score_cap / 115.0;
            out[base + 8] = time_seconds as f32;
            out[base + 9] = z_distance as f32;
        } else {
            out[base] = 0.0;
            out[base + 1] = 0.0;
            out[base + 2] = 0.0;
            out[base + 3] = -1.0;
            out[base + 4] = 0.0;
            out[base + 5] = 0.0;
            out[base + 6] = 0.0;
            out[base + 7] = 0.0;
            out[base + 8] = 0.0;
            out[base + 9] = 0.0;
        }
    }
}

fn build_obstacle_feature_vector(
    out: &mut [f32],
    obstacles: &[BeatmapObstacle],
    obstacle_idx: usize,
    t_beat: f64,
    spawn_ahead_beats: f64,
) {
    for offset in 0..NUM_UPCOMING_OBSTACLES {
        let base = offset * OBSTACLE_FEATURES;
        if obstacle_idx + offset < obstacles.len() {
            let obstacle = &obstacles[obstacle_idx + offset];
            let raw_time_offset = f64::from(obstacle.time) - t_beat;
            if raw_time_offset > spawn_ahead_beats.max(0.0) {
                out[base] = 0.0;
                out[base + 1] = 0.0;
                out[base + 2] = 0.0;
                out[base + 3] = 0.0;
                out[base + 4] = 0.0;
                out[base + 5] = 0.0;
                continue;
            }
            let time_offset = clamp_f64(
                raw_time_offset,
                f64::from(SIM_OBSTACLE_TIME_MIN_BEATS),
                f64::from(SIM_OBSTACLE_TIME_MAX_BEATS),
            );
            out[base] = time_offset as f32;
            out[base + 1] = obstacle.line_index;
            out[base + 2] = obstacle.line_layer;
            out[base + 3] = obstacle.width;
            out[base + 4] = obstacle.height;
            out[base + 5] = obstacle.duration;
        } else {
            out[base] = 0.0;
            out[base + 1] = 0.0;
            out[base + 2] = 0.0;
            out[base + 3] = 0.0;
            out[base + 4] = 0.0;
            out[base + 5] = 0.0;
        }
    }
}

fn frame_velocity(frames: &[PoseFrame], frame_idx: usize) -> [f32; VELOCITY_DIM] {
    if frame_idx == 0 {
        return [0.0_f32; VELOCITY_DIM];
    }

    let prev_idx = frame_idx - 1;
    let dt = (frames[frame_idx].time - frames[prev_idx].time).max(1.0 / 120.0);
    let mut velocity = [0.0_f32; VELOCITY_DIM];
    for axis in 0..VELOCITY_DIM {
        velocity[axis] = (frames[frame_idx].pose[axis] - frames[prev_idx].pose[axis]) / dt;
    }
    velocity
}

fn should_keep_frame(frame_idx: usize, next_delta: f32, prev_delta: f32) -> bool {
    next_delta <= NOTE_LOOKAHEAD_BEATS
        || prev_delta <= FOLLOWTHROUGH_BEATS
        || frame_idx % BACKGROUND_FRAME_STRIDE == 0
}

fn compute_spawn_ahead_beats(bpm: f32, note_jump_speed: f32, note_jump_offset: f32) -> f32 {
    let safe_bpm = bpm.max(1e-6);
    let safe_njs = note_jump_speed.max(0.0);
    let seconds_per_beat = 60.0 / safe_bpm;
    let mut half_jump_beats = INITIAL_HALF_JUMP_BEATS;

    while half_jump_beats > MIN_HALF_JUMP_BEATS
        && (safe_njs * seconds_per_beat * half_jump_beats) > MAX_HALF_JUMP_DISTANCE_METERS
    {
        half_jump_beats *= 0.5;
    }

    (half_jump_beats + note_jump_offset).max(MIN_HALF_JUMP_BEATS)
}

fn fits_float16_slice(slice: &[f32]) -> bool {
    slice
        .iter()
        .all(|value| value.is_finite() && value.abs() <= FLOAT16_MAX)
}

fn resample_frames_to_sim_rate(frames: &[PoseFrame], target_dt: f32) -> Vec<PoseFrame> {
    if frames.len() <= 1 {
        return frames.to_vec();
    }

    let source_start = frames.first().map(|frame| frame.time).unwrap_or(0.0);
    let source_end = frames.last().map(|frame| frame.time).unwrap_or(0.0);
    if source_end <= source_start {
        return vec![frames[0].clone()];
    }

    let stop = source_end + (0.25 * target_dt);
    let sample_times = build_sample_times(source_start, stop, target_dt);
    let sample_times = if sample_times.is_empty() {
        vec![source_start]
    } else {
        sample_times
    };

    let mut resampled = Vec::with_capacity(sample_times.len());
    let mut src_idx = 0_usize;
    let max_src_idx = frames.len() - 1;

    for sample_time in sample_times {
        while src_idx + 1 < frames.len() && frames[src_idx + 1].time < sample_time {
            src_idx += 1;
        }

        let next_idx = (src_idx + 1).min(max_src_idx);
        let left = &frames[src_idx];
        let right = &frames[next_idx];
        let left_time = left.time;
        let right_time = right.time;
        let pose = if next_idx == src_idx || right_time <= left_time {
            left.pose
        } else {
            let alpha = (sample_time - left_time) / (right_time - left_time).max(1e-6);
            interpolate_pose(&left.pose, &right.pose, alpha)
        };

        resampled.push(PoseFrame {
            time: sample_time,
            pose,
        });
    }

    resampled
}

fn build_sample_times(start: f32, stop: f32, step: f32) -> Vec<f32> {
    if !start.is_finite() || !stop.is_finite() || !step.is_finite() || step <= 0.0 {
        return Vec::new();
    }
    let span = f64::from(stop) - f64::from(start);
    if span <= 0.0 {
        return Vec::new();
    }
    let count = (span / f64::from(step)).ceil().max(0.0) as usize;
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        output.push(start + (index as f32) * step);
    }
    output
}

fn interpolate_pose(
    pose_a: &[f32; POSE_DIM],
    pose_b: &[f32; POSE_DIM],
    alpha: f32,
) -> [f32; POSE_DIM] {
    let alpha = alpha.clamp(0.0, 1.0);
    let mut blended = *pose_a;

    for &(start, end) in &[(0_usize, 3_usize), (7, 10), (14, 17)] {
        for idx in start..end {
            blended[idx] = pose_a[idx] + (pose_b[idx] - pose_a[idx]) * alpha;
        }
    }

    for &(start, end) in &[(3_usize, 7_usize), (10, 14), (17, 21)] {
        let mut qa = [
            pose_a[start],
            pose_a[start + 1],
            pose_a[start + 2],
            pose_a[start + 3],
        ];
        let mut qb = [
            pose_b[start],
            pose_b[start + 1],
            pose_b[start + 2],
            pose_b[start + 3],
        ];
        normalize_quaternion_lenient(&mut qa);
        normalize_quaternion_lenient(&mut qb);
        let dot = qa
            .iter()
            .zip(qb.iter())
            .map(|(left, right)| left * right)
            .sum::<f32>();
        if dot < 0.0 {
            for value in &mut qb {
                *value = -*value;
            }
        }
        let mut blended_quat = [0.0_f32; 4];
        for idx in 0..4 {
            blended_quat[idx] = qa[idx] + (qb[idx] - qa[idx]) * alpha;
        }
        normalize_quaternion_lenient(&mut blended_quat);
        blended[start..end].copy_from_slice(&blended_quat);
    }

    blended
}

fn normalize_quaternion_lenient(quat: &mut [f32; 4]) {
    let norm = quat
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    for value in quat.iter_mut() {
        *value /= norm;
    }
}

fn normalize_pose_quaternions(pose: &mut [f32; POSE_DIM]) {
    for &(start, end) in &[(3_usize, 7_usize), (10, 14), (17, 21)] {
        let mut quat = [
            pose[start],
            pose[start + 1],
            pose[start + 2],
            pose[start + 3],
        ];
        normalize_quaternion_lenient(&mut quat);
        pose[start..end].copy_from_slice(&quat);
    }
}

fn simulator_executable_pose_target(
    current_pose: &[f32; POSE_DIM],
    future_pose: &[f32; POSE_DIM],
) -> [f32; POSE_DIM] {
    let horizon = TARGET_POSE_HORIZON_FRAMES.max(1) as f32;
    let mut target_pose = *current_pose;
    for idx in 0..POSE_DIM {
        let target_delta = ((future_pose[idx] - current_pose[idx]) / horizon).clamp(
            -TARGET_ACTION_DELTA_CLAMP[idx],
            TARGET_ACTION_DELTA_CLAMP[idx],
        );
        target_pose[idx] = (current_pose[idx] + target_delta)
            .clamp(-ACTION_ABS_COMPONENT_LIMIT, ACTION_ABS_COMPONENT_LIMIT);
    }
    normalize_pose_quaternions(&mut target_pose);
    target_pose
}

fn assign_split(song_hash: &str) -> &'static str {
    let digest = md5_compute(song_hash.to_ascii_uppercase().as_bytes());
    let threshold = (256.0 * VAL_FRACTION) as u8;
    if digest[0] < threshold {
        "val"
    } else {
        "train"
    }
}

fn write_replay_shard_file(
    split: &str,
    replay_name: &str,
    features: &[f32],
    targets: &[f32],
    sample_count: usize,
    shard_root: &Path,
) -> Result<String> {
    let split_dir = shard_root.join(split);
    fs::create_dir_all(&split_dir)?;
    let shard_id = Path::new(replay_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(replay_name);
    let filename = format!("{shard_id}.safetensors");
    let rel_path = format!("{split}/{filename}");
    let shard_path = split_dir.join(filename);
    let temp_path = shard_path.with_extension(format!("safetensors.tmp.{}", std::process::id()));

    let x_half: Vec<f16> = features.iter().copied().map(f16::from_f32).collect();
    let x_view = TensorView::new(
        Dtype::F16,
        vec![sample_count, INPUT_DIM],
        cast_slice(&x_half),
    )
    .context("failed to construct safetensors view for features")?;
    let y_view = TensorView::new(
        Dtype::F32,
        vec![sample_count, POSE_DIM],
        cast_slice(targets),
    )
    .context("failed to construct safetensors view for targets")?;
    let tensors = BTreeMap::from([("x".to_string(), x_view), ("y".to_string(), y_view)]);
    serialize_to_file(&tensors, &None, &temp_path)
        .with_context(|| format!("failed to write {}", temp_path.display()))?;
    replace_file(&temp_path, &shard_path)?;
    Ok(rel_path.replace('\\', "/"))
}

fn replace_file(temp_path: &Path, final_path: &Path) -> Result<()> {
    if final_path.exists() {
        fs::remove_file(final_path)
            .with_context(|| format!("failed to remove {}", final_path.display()))?;
    }
    fs::rename(temp_path, final_path).with_context(|| {
        format!(
            "failed to move {} -> {}",
            temp_path.display(),
            final_path.display()
        )
    })
}

fn record_manifest_success(manifest: &mut Manifest, payload: ReplaySuccess) {
    if !manifest.done.iter().any(|name| name == &payload.name) {
        manifest.done.push(payload.name.clone());
    }
    manifest.shards.push(ManifestShardRecord {
        split: payload.split.clone(),
        song_hash: payload.replay_meta.song_hash,
        difficulty: payload.replay_meta.difficulty,
        mode: Some(payload.replay_meta.mode),
        bpm: Some(payload.replay_meta.bpm),
        njs: Some(payload.replay_meta.njs),
        jump_offset: Some(payload.replay_meta.jump_offset),
        replay_file: payload.name,
        x_path: None,
        y_path: None,
        shard_path: Some(payload.shard_path),
        tensor_format: Some("safetensors".to_string()),
        samples: payload.sample_count,
        frames_total: payload.replay_meta.frames_total,
        frames_total_raw: payload.replay_meta.frames_total_raw,
        frames_total_resampled: payload.replay_meta.frames_total_resampled,
        sample_hz: payload.replay_meta.sample_hz,
        samples_dropped: payload.replay_meta.samples_dropped,
        note_window_samples: payload.replay_meta.note_window_samples,
        background_samples: payload.replay_meta.background_samples,
        player_id: payload.replay_meta.player_id,
        player_name: payload.replay_meta.player_name,
        replay_timestamp: payload.replay_meta.replay_timestamp,
        game_version: payload.replay_meta.game_version,
        platform: payload.replay_meta.platform,
        tracking_system: payload.replay_meta.tracking_system,
        hmd: payload.replay_meta.hmd,
        controller: payload.replay_meta.controller,
        score: payload.replay_meta.score,
        left_handed: payload.replay_meta.left_handed,
        player_height: payload.replay_meta.player_height,
        replay_start_time: payload.replay_meta.replay_start_time,
        replay_fail_time: payload.replay_meta.replay_fail_time,
    });
    manifest.counts.train_samples += if payload.split == "train" {
        payload.sample_count
    } else {
        0
    };
    manifest.counts.val_samples += if payload.split == "val" {
        payload.sample_count
    } else {
        0
    };
    manifest.counts.train_replays += usize::from(payload.split == "train");
    manifest.counts.val_replays += usize::from(payload.split == "val");
}

fn manifest_semantic_schema() -> ManifestSemanticSchema {
    ManifestSemanticSchema {
        schema_id: MANIFEST_SEMANTIC_SCHEMA_ID.to_string(),
        sample_hz: SIM_SAMPLE_HZ,
        state_history_offsets: STATE_HISTORY_OFFSETS.to_vec(),
        note_feature_layout: NOTE_FEATURE_LAYOUT.to_string(),
        note_lookahead_beats: NOTE_LOOKAHEAD_BEATS,
        followthrough_beats: FOLLOWTHROUGH_BEATS,
        background_frame_stride: BACKGROUND_FRAME_STRIDE,
        target_pose_horizon_frames: TARGET_POSE_HORIZON_FRAMES,
        sim_note_time_range_beats: vec![SIM_NOTE_TIME_MIN_BEATS, SIM_NOTE_TIME_MAX_BEATS],
        note_time_feature_range_beats: vec![
            NOTE_TIME_FEATURE_MIN_BEATS,
            NOTE_TIME_FEATURE_MAX_BEATS,
        ],
        sim_obstacle_time_range_beats: vec![
            SIM_OBSTACLE_TIME_MIN_BEATS,
            SIM_OBSTACLE_TIME_MAX_BEATS,
        ],
        num_upcoming_notes: NUM_UPCOMING_NOTES,
        note_features: NOTE_FEATURES,
        num_upcoming_obstacles: NUM_UPCOMING_OBSTACLES,
        obstacle_features: OBSTACLE_FEATURES,
        pose_dim: POSE_DIM,
        velocity_dim: VELOCITY_DIM,
        track_z_base: TRACK_Z_BASE,
        shard_storage_dtype: SHARD_STORAGE_DTYPE.to_string(),
        feature_storage_dtype: FEATURE_STORAGE_DTYPE.to_string(),
        action_contract: ManifestActionContract {
            action_dim: POSE_DIM,
            action_representation: "absolute_tracked_pose_target".to_string(),
            policy_mean_contract: "current_pose_plus_residual_delta".to_string(),
            simulator_consumption: "GpuBeatSaberSimulator.step(pose_actions)".to_string(),
            absolute_component_limit: ACTION_ABS_COMPONENT_LIMIT,
            per_step_delta_clamp: TARGET_ACTION_DELTA_CLAMP.to_vec(),
            quaternion_slices: vec![vec![3, 7], vec![10, 14], vec![17, 21]],
        },
        target_contract: ManifestTargetContract {
            target_representation: "absolute_tracked_pose_action".to_string(),
            target_generation: "current_pose + clamp((future_pose - current_pose) / target_pose_horizon_frames, -per_step_delta_clamp, per_step_delta_clamp)".to_string(),
            future_pose_horizon_frames: TARGET_POSE_HORIZON_FRAMES,
            normalizes_quaternions: true,
            stored_dtype: TARGET_STORAGE_DTYPE.to_string(),
        },
        sentinel_contract: ManifestSentinelContract {
            missing_note: ManifestMissingNoteSentinel {
                time: 0.0,
                line_index: 0.0,
                line_layer: 0.0,
                note_type: -1.0,
                cut_dx: 0.0,
                cut_dy: 0.0,
                score_class: 0.0,
                score_cap: 0.0,
                time_seconds: 0.0,
                z_distance: 0.0,
            },
            hidden_future_note: "uses_missing_note_sentinel_until_spawn_visible".to_string(),
            missing_obstacle: vec![0.0; OBSTACLE_FEATURES],
        },
        followthrough_contract: ManifestFollowthroughContract {
            keep_next_note_window_beats: NOTE_LOOKAHEAD_BEATS,
            keep_previous_note_window_beats: FOLLOWTHROUGH_BEATS,
            keep_background_frame_stride: BACKGROUND_FRAME_STRIDE,
            timing_reference: "contact_shifted_note_time".to_string(),
        },
        score_class_values: ManifestScoreClassValues {
            normal: SCORE_CLASS_NORMAL,
            arc_head: SCORE_CLASS_ARC_HEAD,
            arc_tail: SCORE_CLASS_ARC_TAIL,
            chain_head: SCORE_CLASS_CHAIN_HEAD,
            chain_link: SCORE_CLASS_CHAIN_LINK,
        },
    }
}

fn manifest_provenance_schema() -> ManifestProvenanceSchema {
    ManifestProvenanceSchema {
        source: "bsor.info".to_string(),
        shard_fields: vec![
            "player_id".to_string(),
            "player_name".to_string(),
            "replay_timestamp".to_string(),
            "game_version".to_string(),
            "platform".to_string(),
            "tracking_system".to_string(),
            "hmd".to_string(),
            "controller".to_string(),
            "score".to_string(),
            "left_handed".to_string(),
            "player_height".to_string(),
            "replay_start_time".to_string(),
            "replay_fail_time".to_string(),
        ],
    }
}

fn manifest_semantic_for_compatibility(manifest: &Manifest) -> Option<ManifestSemanticSchema> {
    if let Some(schema) = manifest.semantic_schema.clone() {
        return Some(schema);
    }
    if manifest.version != MANIFEST_VERSION {
        return None;
    }

    let mut schema = manifest_semantic_schema();
    schema.sample_hz = manifest.sample_hz;
    schema.state_history_offsets = manifest.history_offsets.clone();
    schema.note_feature_layout = manifest.note_feature_layout.clone();
    schema.note_lookahead_beats = manifest.note_lookahead_beats;
    schema.followthrough_beats = manifest.followthrough_beats;
    schema.background_frame_stride = manifest.background_stride;
    schema.target_pose_horizon_frames = manifest.target_pose_horizon_frames;
    Some(schema)
}

fn manifest_compatibility_errors(manifest: &Manifest) -> Vec<String> {
    let mut errors = Vec::new();
    if manifest.version != MANIFEST_VERSION {
        errors.push(format!(
            "version expected {}, got {}",
            MANIFEST_VERSION, manifest.version
        ));
    }
    if manifest.feature_dim != INPUT_DIM {
        errors.push(format!(
            "feature_dim expected {}, got {}",
            INPUT_DIM, manifest.feature_dim
        ));
    }
    if manifest.target_dim != POSE_DIM {
        errors.push(format!(
            "target_dim expected {}, got {}",
            POSE_DIM, manifest.target_dim
        ));
    }
    if manifest.target_pose_horizon_frames != TARGET_POSE_HORIZON_FRAMES {
        errors.push(format!(
            "target_pose_horizon_frames expected {}, got {}",
            TARGET_POSE_HORIZON_FRAMES, manifest.target_pose_horizon_frames
        ));
    }

    match manifest_semantic_for_compatibility(manifest) {
        Some(actual) if actual == manifest_semantic_schema() => {}
        Some(_) => errors.push("semantic_schema mismatch".to_string()),
        None => errors.push("semantic_schema missing".to_string()),
    }
    errors
}

fn set_manifest_warning(manifest: &mut Manifest, warning: ManifestWarning) -> bool {
    let previous = manifest.warnings.clone();
    manifest.warnings.retain(|item| item.code != warning.code);
    manifest.warnings.push(warning);
    manifest.warnings != previous
}

fn clear_manifest_warning(manifest: &mut Manifest, code: &str) -> bool {
    let previous = manifest.warnings.clone();
    manifest.warnings.retain(|item| item.code != code);
    manifest.warnings != previous
}

fn replay_preview(replay_names: &[String], limit: usize) -> String {
    let mut preview: Vec<String> = replay_names.iter().take(limit).cloned().collect();
    if replay_names.len() > limit {
        preview.push("...".to_string());
    }
    preview.join(", ")
}

fn load_manifest(manifest_path: &Path) -> Result<Manifest> {
    if manifest_path.exists() {
        let text = fs::read_to_string(manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?;
        let mut manifest: Manifest = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
        let compatibility_errors = manifest_compatibility_errors(&manifest);
        if compatibility_errors.is_empty() {
            if manifest.provenance_schema.is_none() {
                manifest.provenance_schema = Some(manifest_provenance_schema());
            }
            return Ok(manifest);
        }
        println!("Existing BC shard manifest is incompatible. Rebuilding shards with the current state semantics.");
        for detail in compatibility_errors.iter().take(5) {
            println!("  - {detail}");
        }
    }
    Ok(init_manifest())
}

fn init_manifest() -> Manifest {
    Manifest {
        version: MANIFEST_VERSION,
        feature_dim: INPUT_DIM,
        target_dim: POSE_DIM,
        target_pose_horizon_frames: TARGET_POSE_HORIZON_FRAMES,
        semantic_schema: Some(manifest_semantic_schema()),
        provenance_schema: Some(manifest_provenance_schema()),
        sample_hz: SIM_SAMPLE_HZ,
        history_offsets: STATE_HISTORY_OFFSETS.to_vec(),
        note_feature_layout: NOTE_FEATURE_LAYOUT.to_string(),
        note_lookahead_beats: NOTE_LOOKAHEAD_BEATS,
        followthrough_beats: FOLLOWTHROUGH_BEATS,
        background_stride: BACKGROUND_FRAME_STRIDE,
        shard_format: Some("safetensors".to_string()),
        done: Vec::new(),
        failed: Vec::new(),
        warnings: Vec::new(),
        shards: Vec::new(),
        counts: ManifestCounts::default(),
    }
}

fn save_manifest(manifest: &Manifest, manifest_path: &Path) -> Result<()> {
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp_path = manifest_path.with_extension(format!("json.tmp.{}", std::process::id()));
    let text = serde_json::to_string_pretty(manifest)?;
    fs::write(&temp_path, text)
        .with_context(|| format!("failed to write {}", temp_path.display()))?;
    replace_file(&temp_path, manifest_path)
}

fn load_replay_paths(replay_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(replay_dir)
        .with_context(|| format!("failed to read replay dir {}", replay_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("bsor"))
            .unwrap_or(false)
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn pending_replay_paths(replay_files: Vec<PathBuf>, done_set: &HashSet<String>) -> Vec<PathBuf> {
    replay_files
        .into_iter()
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                return false;
            };
            !done_set.contains(name)
        })
        .collect()
}

fn load_selected_replay_subset(
    selected_scores_path: &Path,
    limit: Option<usize>,
) -> Result<Option<Vec<String>>> {
    let Some(limit) = limit else {
        return Ok(None);
    };
    if limit == 0 {
        return Ok(Some(Vec::new()));
    }
    let text = fs::read_to_string(selected_scores_path).with_context(|| {
        format!(
            "missing selection manifest: {}",
            selected_scores_path.display()
        )
    })?;
    let payload: Value = serde_json::from_str(&text).with_context(|| {
        format!(
            "invalid selection manifest: {}",
            selected_scores_path.display()
        )
    })?;
    let selected = payload
        .get("selected")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            anyhow!(
                "invalid selection manifest: {}",
                selected_scores_path.display()
            )
        })?;
    let mut names = HashSet::new();
    let mut ordered = Vec::new();
    for item in selected {
        let replay_id = item.get("id").and_then(value_to_string).unwrap_or_default();
        if replay_id.is_empty() {
            continue;
        }
        let replay_name = format!("{replay_id}.bsor");
        if names.insert(replay_name.clone()) {
            ordered.push(replay_name);
        }
        if ordered.len() >= limit {
            break;
        }
    }
    Ok(Some(ordered))
}

fn parse_beatmap_dat(dat_content: &[u8]) -> Result<BeatmapData> {
    let data: Value =
        serde_json::from_slice(dat_content).context("failed to parse beatmap json")?;
    let mut notes = Vec::new();
    let mut obstacles = Vec::new();
    let mut arcs = Vec::new();
    let mut chains = Vec::new();

    if let Some(note_list) = data.get("_notes").and_then(Value::as_array) {
        for (index, note) in note_list.iter().enumerate() {
            let note_type = get_i32(note, "_type").unwrap_or(0);
            if matches!(note_type, 0 | 1 | 3) {
                notes.push(make_note(
                    index as i32,
                    get_f32(note, "_time").unwrap_or(0.0),
                    get_f32(note, "_lineIndex").unwrap_or(0.0),
                    get_f32(note, "_lineLayer").unwrap_or(0.0),
                    note_type,
                    get_i32(note, "_cutDirection").unwrap_or(if note_type == 3 { 8 } else { 0 }),
                ));
            }
        }

        for obstacle in data
            .get("_obstacles")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let legacy_type = get_i32(obstacle, "_type").unwrap_or(0);
            let (line_layer, height) = if legacy_type == 1 {
                (2.0, 3.0)
            } else {
                (0.0, 5.0)
            };
            obstacles.push(BeatmapObstacle {
                time: get_f32(obstacle, "_time").unwrap_or(0.0),
                line_index: get_f32(obstacle, "_lineIndex").unwrap_or(0.0),
                line_layer,
                width: get_f32(obstacle, "_width").unwrap_or(1.0),
                height,
                duration: get_f32(obstacle, "_duration").unwrap_or(0.0),
            });
        }

        for slider in data
            .get("_sliders")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            arcs.push(ArcDef {
                color: get_i32(slider, "_colorType").unwrap_or(0),
                head_time: get_f32(slider, "_headTime").unwrap_or(0.0),
                head_line_index: get_f32(slider, "_headLineIndex").unwrap_or(0.0),
                head_line_layer: get_f32(slider, "_headLineLayer").unwrap_or(0.0),
                tail_time: get_f32(slider, "_tailTime").unwrap_or(0.0),
                tail_line_index: get_f32(slider, "_tailLineIndex").unwrap_or(0.0),
                tail_line_layer: get_f32(slider, "_tailLineLayer").unwrap_or(0.0),
            });
        }
    } else if data.get("colorNotes").is_some() || data.get("bombNotes").is_some() {
        if let Some(color_notes) = data.get("colorNotes").and_then(Value::as_array) {
            for note in color_notes {
                let index = notes.len() as i32;
                notes.push(make_note(
                    index,
                    get_f32(note, "b").unwrap_or(0.0),
                    get_f32(note, "x").unwrap_or(0.0),
                    get_f32(note, "y").unwrap_or(0.0),
                    get_i32(note, "c").unwrap_or(0),
                    get_i32(note, "d").unwrap_or(0),
                ));
            }
        }
        if let Some(bomb_notes) = data.get("bombNotes").and_then(Value::as_array) {
            for note in bomb_notes {
                let index = notes.len() as i32;
                notes.push(make_note(
                    index,
                    get_f32(note, "b").unwrap_or(0.0),
                    get_f32(note, "x").unwrap_or(0.0),
                    get_f32(note, "y").unwrap_or(0.0),
                    3,
                    8,
                ));
            }
        }
        for obstacle in data
            .get("obstacles")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            obstacles.push(BeatmapObstacle {
                time: get_f32(obstacle, "b").unwrap_or(0.0),
                line_index: get_f32(obstacle, "x").unwrap_or(0.0),
                line_layer: get_f32(obstacle, "y").unwrap_or(0.0),
                width: get_f32(obstacle, "w").unwrap_or(1.0),
                height: get_f32(obstacle, "h").unwrap_or(5.0),
                duration: get_f32(obstacle, "d").unwrap_or(0.0),
            });
        }
        for slider in data
            .get("sliders")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            arcs.push(ArcDef {
                color: get_i32(slider, "c").unwrap_or(0),
                head_time: get_f32(slider, "b").unwrap_or(0.0),
                head_line_index: get_f32(slider, "x").unwrap_or(0.0),
                head_line_layer: get_f32(slider, "y").unwrap_or(0.0),
                tail_time: get_f32(slider, "tb").unwrap_or(0.0),
                tail_line_index: get_f32(slider, "tx").unwrap_or(0.0),
                tail_line_layer: get_f32(slider, "ty").unwrap_or(0.0),
            });
        }
        for chain in data
            .get("burstSliders")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            chains.push(ChainDef {
                color: get_i32(chain, "c").unwrap_or(0),
                head_time: get_f32(chain, "b").unwrap_or(0.0),
                head_line_index: get_f32(chain, "x").unwrap_or(0.0),
                head_line_layer: get_f32(chain, "y").unwrap_or(0.0),
                tail_time: get_f32(chain, "tb").unwrap_or(0.0),
                tail_line_index: get_f32(chain, "tx").unwrap_or(0.0),
                tail_line_layer: get_f32(chain, "ty").unwrap_or(0.0),
                slice_count: get_usize(chain, "sc").unwrap_or(1).max(1),
                squish: get_f32(chain, "s").unwrap_or(1.0),
            });
        }
    }

    notes.sort_by(|left, right| float_cmp(left.time, right.time));
    obstacles.sort_by(|left, right| float_cmp(left.time, right.time));

    let mut note_lookup: HashMap<NoteKey, Vec<usize>> = HashMap::new();
    for (idx, note) in notes.iter_mut().enumerate() {
        note.index = idx as i32;
        if matches!(note.note_type, 0 | 1) {
            let key = match_note_key(note.time, note.line_index, note.line_layer, note.note_type);
            note_lookup.entry(key).or_default().push(idx);
        }
    }

    for arc in &arcs {
        let head_key = match_note_key(
            arc.head_time,
            arc.head_line_index,
            arc.head_line_layer,
            arc.color,
        );
        let tail_key = match_note_key(
            arc.tail_time,
            arc.tail_line_index,
            arc.tail_line_layer,
            arc.color,
        );
        if let Some(indices) = note_lookup.get(&head_key) {
            for &index in indices {
                notes[index].arc_head = true;
            }
        }
        if let Some(indices) = note_lookup.get(&tail_key) {
            for &index in indices {
                notes[index].arc_tail = true;
            }
        }
    }

    let mut next_index = notes.len() as i32;
    for chain in &chains {
        let head_key = match_note_key(
            chain.head_time,
            chain.head_line_index,
            chain.head_line_layer,
            chain.color,
        );
        if let Some(indices) = note_lookup.get(&head_key) {
            for &index in indices {
                notes[index].chain_head = true;
            }
        }

        let slice_count = chain.slice_count.max(1);
        let squish = chain.squish.max(1e-3);
        for slice_idx in 1..slice_count {
            let frac = (((slice_idx as f32) / ((slice_count - 1).max(1) as f32)) * squish).min(1.0);
            let mut link = make_note(
                next_index,
                chain.head_time + (chain.tail_time - chain.head_time) * frac,
                chain.head_line_index + (chain.tail_line_index - chain.head_line_index) * frac,
                chain.head_line_layer + (chain.tail_line_layer - chain.head_line_layer) * frac,
                chain.color,
                8,
            );
            link.chain_link = true;
            notes.push(link);
            next_index += 1;
        }
    }

    notes.sort_by(|left, right| {
        float_cmp(left.time, right.time)
            .then_with(|| left.note_type.cmp(&right.note_type))
            .then_with(|| float_cmp(left.line_index, right.line_index))
            .then_with(|| float_cmp(left.line_layer, right.line_layer))
    });
    for (idx, note) in notes.iter_mut().enumerate() {
        note.index = idx as i32;
        finalize_note_scoring(note);
    }

    Ok(BeatmapData {
        notes,
        obstacles,
        mode: "Standard".to_string(),
        difficulty: None,
        njs: 18.0,
        offset: 0.0,
        song_name: String::new(),
        song_author_name: String::new(),
        level_author_name: String::new(),
        environment_name: "DefaultEnvironment".to_string(),
    })
}

fn make_note(
    index: i32,
    time: f32,
    line_index: f32,
    line_layer: f32,
    note_type: i32,
    cut_direction: i32,
) -> BeatmapNote {
    BeatmapNote {
        index,
        time,
        line_index,
        line_layer,
        note_type,
        cut_direction,
        arc_head: false,
        arc_tail: false,
        chain_head: false,
        chain_link: false,
        score_class: SCORE_CLASS_NORMAL,
        score_cap: 115.0,
    }
}

fn finalize_note_scoring(note: &mut BeatmapNote) {
    if note.note_type == 3 {
        note.score_class = -1.0;
        note.score_cap = 0.0;
        return;
    }

    if note.chain_link {
        note.score_class = SCORE_CLASS_CHAIN_LINK;
        note.score_cap = 20.0;
        note.cut_direction = 8;
        return;
    }

    if note.chain_head {
        note.score_class = SCORE_CLASS_CHAIN_HEAD;
        note.score_cap = 85.0;
    } else if note.arc_head {
        note.score_class = SCORE_CLASS_ARC_HEAD;
    } else if note.arc_tail {
        note.score_class = SCORE_CLASS_ARC_TAIL;
    }
}

fn match_note_key(time: f32, line_index: f32, line_layer: f32, note_type: i32) -> NoteKey {
    NoteKey {
        time_micros: round_ties_even_scaled(time, 1_000_000.0),
        line_index_e4: round_ties_even_scaled(line_index, 10_000.0),
        line_layer_e4: round_ties_even_scaled(line_layer, 10_000.0),
        note_type,
    }
}

fn round_ties_even_scaled(value: f32, scale: f64) -> i64 {
    (f64::from(value) * scale).round_ties_even() as i64
}

fn collect_difficulty_entries(info_data: &Value) -> Vec<DifficultyEntry> {
    let mut entries = Vec::new();

    if let Some(sets) = info_data
        .get("_difficultyBeatmapSets")
        .and_then(Value::as_array)
    {
        for set_data in sets {
            let mode = get_string(set_data, "_beatmapCharacteristicName")
                .unwrap_or_else(|| "Standard".to_string());
            if let Some(beatmaps) = set_data
                .get("_difficultyBeatmaps")
                .and_then(Value::as_array)
            {
                for beatmap in beatmaps {
                    let difficulty = get_string(beatmap, "_difficulty").unwrap_or_default();
                    if let Some(filename) = get_string(beatmap, "_beatmapFilename") {
                        entries.push(DifficultyEntry {
                            mode: mode.clone(),
                            difficulty: difficulty.clone(),
                            filename,
                            rank: difficulty_rank(&difficulty),
                            note_jump_movement_speed: get_f32(beatmap, "_noteJumpMovementSpeed"),
                            note_jump_start_beat_offset: get_f32(
                                beatmap,
                                "_noteJumpStartBeatOffset",
                            ),
                        });
                    }
                }
            }
        }
    }

    if let Some(sets) = info_data
        .get("difficultyBeatmapSets")
        .and_then(Value::as_array)
    {
        for set_data in sets {
            let mode = get_string(set_data, "beatmapCharacteristicName")
                .unwrap_or_else(|| "Standard".to_string());
            if let Some(beatmaps) = set_data.get("difficultyBeatmaps").and_then(Value::as_array) {
                for beatmap in beatmaps {
                    let difficulty = get_string_any(
                        beatmap,
                        &["difficulty", "difficultyName", "customDifficultyName"],
                    )
                    .unwrap_or_default();
                    if let Some(filename) = get_string(beatmap, "beatmapFilename") {
                        entries.push(DifficultyEntry {
                            mode: mode.clone(),
                            difficulty: difficulty.clone(),
                            filename,
                            rank: difficulty_rank(&difficulty),
                            note_jump_movement_speed: get_f32(beatmap, "noteJumpMovementSpeed"),
                            note_jump_start_beat_offset: get_f32(
                                beatmap,
                                "noteJumpStartBeatOffset",
                            ),
                        });
                    }
                }
            }
        }
    }

    entries
}

fn select_difficulty_entry(
    entries: &[DifficultyEntry],
    preferred_mode: Option<&str>,
    preferred_difficulty: Option<&str>,
    strict_mode: bool,
) -> Option<DifficultyEntry> {
    if entries.is_empty() {
        return None;
    }
    let pref_mode = normalize_mode_name(preferred_mode.unwrap_or("Standard"));
    let pref_diff = normalize_difficulty_name(preferred_difficulty.unwrap_or(""));

    if let Some(found) = entries.iter().find(|entry| {
        normalize_mode_name(&entry.mode) == pref_mode
            && normalize_difficulty_name(&entry.difficulty) == pref_diff
    }) {
        return Some(found.clone());
    }

    if !pref_diff.is_empty() {
        return None;
    }

    let mode_matches: Vec<_> = entries
        .iter()
        .filter(|entry| normalize_mode_name(&entry.mode) == pref_mode)
        .cloned()
        .collect();
    if !mode_matches.is_empty() {
        return mode_matches
            .into_iter()
            .max_by(|left, right| left.rank.cmp(&right.rank));
    }

    if strict_mode && preferred_mode.is_some() {
        return None;
    }

    let standard_matches: Vec<_> = entries
        .iter()
        .filter(|entry| normalize_mode_name(&entry.mode) == "standard")
        .cloned()
        .collect();
    if !standard_matches.is_empty() {
        return standard_matches
            .into_iter()
            .max_by(|left, right| left.rank.cmp(&right.rank));
    }

    entries
        .iter()
        .cloned()
        .max_by(|left, right| left.rank.cmp(&right.rank))
}

fn difficulty_rank(difficulty_name: &str) -> i32 {
    match normalize_difficulty_name(difficulty_name).as_str() {
        "easy" => 0,
        "normal" => 1,
        "hard" => 2,
        "expert" => 3,
        "expertplus" => 4,
        _ => -1,
    }
}

fn normalize_token(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn normalize_mode_name(mode_name: &str) -> String {
    match normalize_token(mode_name).as_str() {
        "" | "standard" => "standard".to_string(),
        "360degree" | "360" => "360degree".to_string(),
        "90degree" | "90" => "90degree".to_string(),
        "onesaber" | "onesaberstandard" => "onesaber".to_string(),
        "noarrows" => "noarrows".to_string(),
        other => other.to_string(),
    }
}

fn normalize_difficulty_name(difficulty_name: &str) -> String {
    match normalize_token(difficulty_name).as_str() {
        "expertplus" => "expertplus".to_string(),
        "expert" => "expert".to_string(),
        "hard" => "hard".to_string(),
        "normal" => "normal".to_string(),
        "easy" => "easy".to_string(),
        other => other.to_string(),
    }
}

fn resolve_map_zip_path(maps_dir: &Path, map_hash: &str) -> Option<PathBuf> {
    let candidates = [
        maps_dir.join(format!("{map_hash}.zip")),
        maps_dir.join(format!("{}.zip", map_hash.to_ascii_uppercase())),
        maps_dir.join(format!("{}.zip", map_hash.to_ascii_lowercase())),
    ];
    candidates.into_iter().find(|path| path.exists())
}

fn find_info_file_name<R: Read + Seek>(archive: &mut ZipArchive<R>) -> Result<Option<String>> {
    for index in 0..archive.len() {
        let file = archive.by_index(index)?;
        if Path::new(file.name())
            .file_name()
            .and_then(|value| value.to_str())
            .map(|name| name.eq_ignore_ascii_case("info.dat"))
            .unwrap_or(false)
        {
            return Ok(Some(file.name().to_string()));
        }
    }
    Ok(None)
}

fn find_first_dat_file<R: Read + Seek>(archive: &mut ZipArchive<R>) -> Result<Option<String>> {
    for index in 0..archive.len() {
        let file = archive.by_index(index)?;
        let lower = file.name().to_ascii_lowercase();
        if lower.ends_with(".dat")
            && Path::new(&lower)
                .file_name()
                .and_then(|value| value.to_str())
                != Some("info.dat")
        {
            return Ok(Some(file.name().to_string()));
        }
    }
    Ok(None)
}

fn resolve_zip_entry_name<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    wanted: &str,
) -> Option<String> {
    let wanted_lower = wanted.replace('\\', "/").to_ascii_lowercase();
    for index in 0..archive.len() {
        let file = archive.by_index(index).ok()?;
        let name = file.name().replace('\\', "/");
        if name.to_ascii_lowercase() == wanted_lower {
            return Some(file.name().to_string());
        }
    }

    let wanted_basename = Path::new(wanted)
        .file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())?;
    for index in 0..archive.len() {
        let file = archive.by_index(index).ok()?;
        let basename = Path::new(file.name())
            .file_name()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase());
        if basename.as_deref() == Some(wanted_basename.as_str()) {
            return Some(file.name().to_string());
        }
    }
    None
}

fn read_zip_entry_bytes<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<Vec<u8>> {
    let mut file = archive
        .by_name(name)
        .with_context(|| format!("missing zip entry {}", name))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn get_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(value_to_string)
}

fn get_string_any(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| get_string(value, key))
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn format_count(value: usize) -> String {
    let text = value.to_string();
    let mut formatted = String::with_capacity(text.len() + (text.len() / 3));
    let chars: Vec<_> = text.chars().collect();
    for (index, ch) in chars.iter().enumerate() {
        if index > 0 && (chars.len() - index) % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(*ch);
    }
    formatted
}

fn get_f32(value: &Value, key: &str) -> Option<f32> {
    value.get(key).and_then(value_to_f32)
}

fn get_f32_any(value: &Value, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| get_f32(value, key))
}

fn value_to_f32(value: &Value) -> Option<f32> {
    match value {
        Value::Number(number) => number.as_f64().map(|value| value as f32),
        Value::String(text) => text.parse::<f32>().ok(),
        _ => None,
    }
}

fn get_i32(value: &Value, key: &str) -> Option<i32> {
    value.get(key).and_then(|inner| match inner {
        Value::Number(number) => number.as_i64().map(|value| value as i32),
        Value::String(text) => text.parse::<i32>().ok(),
        _ => None,
    })
}

fn get_usize(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(|inner| match inner {
        Value::Number(number) => number.as_u64().map(|value| value as usize),
        Value::String(text) => text.parse::<usize>().ok(),
        _ => None,
    })
}

fn clamp_f64(value: f64, min_value: f64, max_value: f64) -> f64 {
    value.max(min_value).min(max_value)
}

fn float_cmp(left: f32, right: f32) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn encode_cut_direction(cut_dir: i32) -> (f32, f32) {
    match cut_dir {
        0 => (0.0, 1.0),
        1 => (0.0, -1.0),
        2 => (-1.0, 0.0),
        3 => (1.0, 0.0),
        4 => (-0.7071, 0.7071),
        5 => (0.7071, 0.7071),
        6 => (-0.7071, -0.7071),
        7 => (0.7071, -0.7071),
        _ => (0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    use zip::write::SimpleFileOptions;

    use crate::io::write_bsor_path;
    use crate::model::{Bsor, Info, MAGIC_NUMBER};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cybernoodles_{name}_{}_{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_expertplus_only_map(maps_dir: &Path, map_hash: &str) -> Result<()> {
        let zip_path = maps_dir.join(format!("{map_hash}.zip"));
        let file = File::create(zip_path)?;
        let mut zip = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default();
        zip.start_file("Info.dat", options)?;
        zip.write_all(
            br#"{
                "_version": "2.1.0",
                "_songName": "Difficulty Contract Test",
                "_beatsPerMinute": 120.0,
                "_difficultyBeatmapSets": [{
                    "_beatmapCharacteristicName": "Standard",
                    "_difficultyBeatmaps": [{
                        "_difficulty": "ExpertPlus",
                        "_beatmapFilename": "ExpertPlusStandard.dat",
                        "_noteJumpMovementSpeed": 18.0,
                        "_noteJumpStartBeatOffset": 0.0
                    }]
                }]
            }"#,
        )?;
        zip.start_file("ExpertPlusStandard.dat", options)?;
        zip.write_all(br#"{"_version":"2.0.0","_notes":[],"_obstacles":[]}"#)?;
        zip.finish()?;
        Ok(())
    }

    fn test_replay(song_hash: &str, difficulty: &str) -> Bsor {
        Bsor {
            magic_number: MAGIC_NUMBER,
            file_version: 1,
            info: Info {
                version: "1.0.0".to_string(),
                game_version: "1.0.0".to_string(),
                timestamp: "0".to_string(),
                player_id: "player".to_string(),
                player_name: "Player".to_string(),
                platform: "pc".to_string(),
                tracking_system: "openvr".to_string(),
                hmd: "hmd".to_string(),
                controller: "controller".to_string(),
                song_hash: song_hash.to_string(),
                song_name: "Song".to_string(),
                mapper: "Mapper".to_string(),
                difficulty: difficulty.to_string(),
                score: 0,
                mode: "Standard".to_string(),
                environment: "DefaultEnvironment".to_string(),
                modifiers: String::new(),
                jump_distance: 0.0,
                left_handed: false,
                height: 1.7,
                start_time: 0.0,
                fail_time: 0.0,
                speed: 1.0,
            },
            frames: Vec::new(),
            notes: Vec::new(),
            walls: Vec::new(),
            heights: Vec::new(),
            pauses: Vec::new(),
            controller_offsets: None,
            user_data: Vec::new(),
        }
    }

    #[test]
    fn explicit_replay_difficulty_rejects_wrong_chart() -> Result<()> {
        let root = unique_temp_dir("explicit_replay_difficulty");
        let maps_dir = root.join("maps");
        fs::create_dir_all(&maps_dir)?;
        let replay_path = root.join("hard.bsor");
        let map_hash = "DIFFICULTYCONTRACT";
        write_expertplus_only_map(&maps_dir, map_hash)?;
        write_bsor_path(&replay_path, &test_replay(map_hash, "Hard"))?;

        let map_cache = MapCache::new(maps_dir);
        let err = process_single(&replay_path, &map_cache).unwrap_err();

        assert!(format!("{err:#}").contains("missing map/difficulty"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn no_explicit_difficulty_selects_hardest_standard_truthfully() -> Result<()> {
        let root = unique_temp_dir("generic_difficulty");
        let maps_dir = root.join("maps");
        fs::create_dir_all(&maps_dir)?;
        let map_hash = "GENERICCONTRACT";
        write_expertplus_only_map(&maps_dir, map_hash)?;

        let map_cache = MapCache::new(maps_dir);
        let resolved = map_cache
            .get_map_data(map_hash, None, Some("Standard"))?
            .expect("generic Standard request should select hardest Standard chart");

        assert_eq!(resolved.beatmap.mode, "Standard");
        assert_eq!(resolved.beatmap.difficulty.as_deref(), Some("ExpertPlus"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn failed_manifest_entries_remain_pending_for_retry() {
        let replay_files = vec![
            PathBuf::from("already_done.bsor"),
            PathBuf::from("failed_until_map_downloaded.bsor"),
        ];
        let done_set = HashSet::from(["already_done.bsor".to_string()]);

        let pending = pending_replay_paths(replay_files, &done_set);

        assert_eq!(
            pending,
            vec![PathBuf::from("failed_until_map_downloaded.bsor")]
        );
    }
}
