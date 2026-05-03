use serde::Serialize;

use crate::model::Bsor;

const DEFAULT_POSE: [f32; 21] = [
    0.0, 1.55, 0.0, 0.0, 0.0, 0.0, 1.0, -0.30, 1.15, 0.35, 0.0, 0.0, 0.0, 1.0, 0.30, 1.15, 0.35,
    0.0, 0.0, 0.0, 1.0,
];
const POSITION_ABS_MAX: f32 = 8.0;
const QUAT_COMPONENT_ABS_MAX: f32 = 2.0;

const POSE_POSITION_SLICES: [(usize, usize); 3] = [(0, 3), (7, 10), (14, 17)];
const POSE_QUATERNION_SLICES: [(usize, usize); 3] = [(3, 7), (10, 14), (17, 21)];

#[derive(Debug, Clone, Serialize)]
pub struct DatasetFrame {
    pub time: f32,
    pub pose: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayMeta {
    pub song_hash: Option<String>,
    pub difficulty: Option<String>,
    pub mode: String,
    pub modifiers: String,
    pub sanitized_time_frames: u32,
    pub sanitized_pose_segments: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetView {
    pub frames: Vec<DatasetFrame>,
    pub meta: ReplayMeta,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    pub frame_count: usize,
    pub note_count: usize,
    pub wall_count: usize,
    pub pause_count: usize,
    pub user_data_count: usize,
    pub left_span: f32,
    pub right_span: f32,
    pub song_hash: String,
    pub difficulty: String,
    pub mode: String,
}

fn sanitize_frame_time(raw_time: f32, prev_time: Option<f32>) -> (f32, bool) {
    let mut value = raw_time;
    let mut sanitized = false;
    let fallback = prev_time.map(|time| time + (1.0 / 120.0)).unwrap_or(0.0);

    if !value.is_finite() {
        value = fallback;
        sanitized = true;
    }
    if let Some(prev) = prev_time {
        if value < prev {
            value = prev;
            sanitized = true;
        }
    }
    (value, sanitized)
}

fn normalize_quaternion(segment: &mut [f32]) -> bool {
    let mut norm_sq = 0.0_f32;
    for value in segment.iter() {
        norm_sq += value * value;
    }
    let norm = norm_sq.sqrt();
    if !norm.is_finite() || norm < 1e-6 {
        return false;
    }
    for value in segment.iter_mut() {
        *value /= norm;
    }
    true
}

fn pose_frame(frame: &crate::model::Frame) -> [f32; 21] {
    [
        frame.head.x,
        frame.head.y,
        frame.head.z,
        frame.head.x_rot,
        frame.head.y_rot,
        frame.head.z_rot,
        frame.head.w_rot,
        frame.left_hand.x,
        frame.left_hand.y,
        frame.left_hand.z,
        frame.left_hand.x_rot,
        frame.left_hand.y_rot,
        frame.left_hand.z_rot,
        frame.left_hand.w_rot,
        frame.right_hand.x,
        frame.right_hand.y,
        frame.right_hand.z,
        frame.right_hand.x_rot,
        frame.right_hand.y_rot,
        frame.right_hand.z_rot,
        frame.right_hand.w_rot,
    ]
}

fn sanitize_pose(raw_pose: [f32; 21], prev_pose: Option<&[f32; 21]>) -> ([f32; 21], u32) {
    let mut pose = raw_pose;
    let fallback = prev_pose.unwrap_or(&DEFAULT_POSE);
    let mut replaced_segments = 0_u32;

    for (start, end) in POSE_POSITION_SLICES {
        let segment = &pose[start..end];
        let finite = segment.iter().all(|value| value.is_finite());
        let max_abs = segment
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, |acc, value| acc.max(value));
        if !finite || max_abs > POSITION_ABS_MAX {
            pose[start..end].copy_from_slice(&fallback[start..end]);
            replaced_segments += 1;
        }
    }

    for (start, end) in POSE_QUATERNION_SLICES {
        let finite = pose[start..end].iter().all(|value| value.is_finite());
        let max_abs = pose[start..end]
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, |acc, value| acc.max(value));
        if !finite || max_abs > QUAT_COMPONENT_ABS_MAX {
            pose[start..end].copy_from_slice(&fallback[start..end]);
            replaced_segments += 1;
            continue;
        }
        if !normalize_quaternion(&mut pose[start..end]) {
            pose[start..end].copy_from_slice(&fallback[start..end]);
            replaced_segments += 1;
        }
    }

    (pose, replaced_segments)
}

pub fn dataset_view(replay: &Bsor) -> DatasetView {
    let mut prev_time = None;
    let mut prev_pose = None;
    let mut frames = Vec::with_capacity(replay.frames.len());
    let mut sanitized_time_frames = 0_u32;
    let mut sanitized_pose_segments = 0_u32;

    for frame in &replay.frames {
        let (time, time_sanitized) = sanitize_frame_time(frame.time, prev_time);
        let (pose, replaced_segments) = sanitize_pose(pose_frame(frame), prev_pose.as_ref());
        frames.push(DatasetFrame {
            time,
            pose: pose.to_vec(),
        });
        prev_time = Some(time);
        prev_pose = Some(pose);
        sanitized_time_frames += u32::from(time_sanitized);
        sanitized_pose_segments += replaced_segments;
    }

    DatasetView {
        frames,
        meta: ReplayMeta {
            song_hash: if replay.info.song_hash.is_empty() {
                None
            } else {
                Some(replay.info.song_hash.clone())
            },
            difficulty: if replay.info.difficulty.is_empty() {
                None
            } else {
                Some(replay.info.difficulty.clone())
            },
            mode: if replay.info.mode.is_empty() {
                "Standard".to_string()
            } else {
                replay.info.mode.clone()
            },
            modifiers: replay.info.modifiers.clone(),
            sanitized_time_frames,
            sanitized_pose_segments,
        },
    }
}

pub fn validation_summary(replay: &Bsor) -> ValidationSummary {
    let mut left_min = [f32::INFINITY; 3];
    let mut left_max = [f32::NEG_INFINITY; 3];
    let mut right_min = [f32::INFINITY; 3];
    let mut right_max = [f32::NEG_INFINITY; 3];

    for frame in &replay.frames {
        let left = [frame.left_hand.x, frame.left_hand.y, frame.left_hand.z];
        let right = [frame.right_hand.x, frame.right_hand.y, frame.right_hand.z];
        for axis in 0..3 {
            left_min[axis] = left_min[axis].min(left[axis]);
            left_max[axis] = left_max[axis].max(left[axis]);
            right_min[axis] = right_min[axis].min(right[axis]);
            right_max[axis] = right_max[axis].max(right[axis]);
        }
    }

    let left_span = if replay.frames.is_empty() {
        0.0
    } else {
        ((left_max[0] - left_min[0]).powi(2)
            + (left_max[1] - left_min[1]).powi(2)
            + (left_max[2] - left_min[2]).powi(2))
        .sqrt()
    };
    let right_span = if replay.frames.is_empty() {
        0.0
    } else {
        ((right_max[0] - right_min[0]).powi(2)
            + (right_max[1] - right_min[1]).powi(2)
            + (right_max[2] - right_min[2]).powi(2))
        .sqrt()
    };

    ValidationSummary {
        frame_count: replay.frames.len(),
        note_count: replay.notes.len(),
        wall_count: replay.walls.len(),
        pause_count: replay.pauses.len(),
        user_data_count: replay.user_data.len(),
        left_span,
        right_span,
        song_hash: replay.info.song_hash.clone(),
        difficulty: replay.info.difficulty.clone(),
        mode: replay.info.mode.clone(),
    }
}
