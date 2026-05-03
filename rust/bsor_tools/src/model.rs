use serde::{Deserialize, Serialize};

pub const MAGIC_NUMBER: u32 = 0x442d3d69;
pub const MAX_SUPPORTED_VERSION: u8 = 1;

pub const NOTE_EVENT_GOOD: u32 = 0;
pub const NOTE_EVENT_BAD: u32 = 1;
pub const NOTE_EVENT_MISS: u32 = 2;
pub const NOTE_EVENT_BOMB: u32 = 3;

pub const NOTE_SCORE_TYPE_NORMAL_1: u32 = 0;
pub const NOTE_SCORE_TYPE_IGNORE: u32 = 1;
pub const NOTE_SCORE_TYPE_NOSCORE: u32 = 2;
pub const NOTE_SCORE_TYPE_NORMAL_2: u32 = 3;
pub const NOTE_SCORE_TYPE_SLIDERHEAD: u32 = 4;
pub const NOTE_SCORE_TYPE_SLIDERTAIL: u32 = 5;
pub const NOTE_SCORE_TYPE_BURSTSLIDERHEAD: u32 = 6;
pub const NOTE_SCORE_TYPE_BURSTSLIDERELEMENT: u32 = 7;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Info {
    pub version: String,
    #[serde(rename = "gameVersion")]
    pub game_version: String,
    pub timestamp: String,
    #[serde(rename = "playerId")]
    pub player_id: String,
    #[serde(rename = "playerName")]
    pub player_name: String,
    pub platform: String,
    #[serde(rename = "trackingSystem")]
    pub tracking_system: String,
    pub hmd: String,
    pub controller: String,
    #[serde(rename = "songHash")]
    pub song_hash: String,
    #[serde(rename = "songName")]
    pub song_name: String,
    pub mapper: String,
    pub difficulty: String,
    pub score: u32,
    pub mode: String,
    pub environment: String,
    pub modifiers: String,
    #[serde(rename = "jumpDistance")]
    pub jump_distance: f32,
    #[serde(rename = "leftHanded")]
    pub left_handed: bool,
    pub height: f32,
    #[serde(rename = "startTime")]
    pub start_time: f32,
    #[serde(rename = "failTime")]
    pub fail_time: f32,
    pub speed: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrObject {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    #[serde(rename = "x_rot")]
    pub x_rot: f32,
    #[serde(rename = "y_rot")]
    pub y_rot: f32,
    #[serde(rename = "z_rot")]
    pub z_rot: f32,
    #[serde(rename = "w_rot")]
    pub w_rot: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub time: f32,
    pub fps: u32,
    pub head: VrObject,
    #[serde(rename = "left_hand")]
    pub left_hand: VrObject,
    #[serde(rename = "right_hand")]
    pub right_hand: VrObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cut {
    #[serde(rename = "speedOK")]
    pub speed_ok: bool,
    #[serde(rename = "directionOk")]
    pub direction_ok: bool,
    #[serde(rename = "saberTypeOk")]
    pub saber_type_ok: bool,
    #[serde(rename = "wasCutTooSoon")]
    pub was_cut_too_soon: bool,
    #[serde(rename = "saberSpeed")]
    pub saber_speed: f32,
    #[serde(rename = "saberDirection")]
    pub saber_direction: [f32; 3],
    #[serde(rename = "saberType")]
    pub saber_type: u32,
    #[serde(rename = "timeDeviation")]
    pub time_deviation: f32,
    #[serde(rename = "cutDeviation")]
    pub cut_deviation: f32,
    #[serde(rename = "cutPoint")]
    pub cut_point: [f32; 3],
    #[serde(rename = "cutNormal")]
    pub cut_normal: [f32; 3],
    #[serde(rename = "cutDistanceToCenter")]
    pub cut_distance_to_center: f32,
    #[serde(rename = "cutAngle")]
    pub cut_angle: f32,
    #[serde(rename = "beforeCutRating")]
    pub before_cut_rating: f32,
    #[serde(rename = "afterCutRating")]
    pub after_cut_rating: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    #[serde(rename = "note_id")]
    pub note_id: u32,
    #[serde(rename = "scoringType", default)]
    pub scoring_type: u32,
    #[serde(rename = "lineIndex", default)]
    pub line_index: u32,
    #[serde(rename = "noteLineLayer", default)]
    pub note_line_layer: u32,
    #[serde(rename = "colorType", default)]
    pub color_type: u32,
    #[serde(rename = "cutDirection", default)]
    pub cut_direction: u32,
    #[serde(rename = "event_time")]
    pub event_time: f32,
    #[serde(rename = "spawn_time")]
    pub spawn_time: f32,
    #[serde(rename = "event_type")]
    pub event_type: u32,
    #[serde(default)]
    pub cut: Option<Cut>,
    #[serde(rename = "pre_score", default)]
    pub pre_score: u32,
    #[serde(rename = "post_score", default)]
    pub post_score: u32,
    #[serde(rename = "acc_score", default)]
    pub acc_score: u32,
    #[serde(default)]
    pub score: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wall {
    pub id: u32,
    pub energy: f32,
    pub time: f32,
    #[serde(rename = "spawnTime")]
    pub spawn_time: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Height {
    pub height: f32,
    pub time: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pause {
    pub duration: u64,
    pub time: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerOffsets {
    pub left: VrObject,
    pub right: VrObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserData {
    pub key: String,
    #[serde(rename = "bytes_base64", with = "crate::model::base64_bytes")]
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bsor {
    pub magic_number: u32,
    pub file_version: u8,
    pub info: Info,
    pub frames: Vec<Frame>,
    pub notes: Vec<Note>,
    pub walls: Vec<Wall>,
    pub heights: Vec<Height>,
    pub pauses: Vec<Pause>,
    pub controller_offsets: Option<ControllerOffsets>,
    #[serde(default)]
    pub user_data: Vec<UserData>,
}

impl Note {
    pub fn refresh_derived_fields(&mut self) {
        let (scoring_type, line_index, note_line_layer, color_type, cut_direction) =
            decode_note_id(self.note_id);
        self.scoring_type = scoring_type;
        self.line_index = line_index;
        self.note_line_layer = note_line_layer;
        self.color_type = color_type;
        self.cut_direction = cut_direction;

        if matches!(self.event_type, NOTE_EVENT_GOOD | NOTE_EVENT_BAD) {
            if let Some(cut) = &self.cut {
                let (pre_score, post_score, acc_score) = calc_note_score(cut, self.scoring_type);
                self.pre_score = pre_score;
                self.post_score = post_score;
                self.acc_score = acc_score;
                self.score = pre_score + post_score + acc_score;
            } else {
                self.pre_score = 0;
                self.post_score = 0;
                self.acc_score = 0;
                self.score = 0;
            }
        } else {
            self.pre_score = 0;
            self.post_score = 0;
            self.acc_score = 0;
            self.score = 0;
        }
    }
}

pub fn decode_note_id(note_id: u32) -> (u32, u32, u32, u32, u32) {
    let mut x = note_id;
    let cut_direction = x % 10;
    x = (x - cut_direction) / 10;
    let color_type = x % 10;
    x = (x - color_type) / 10;
    let note_line_layer = x % 10;
    x = (x - note_line_layer) / 10;
    let line_index = x % 10;
    x = (x - line_index) / 10;
    let scoring_type = x % 10;
    (
        scoring_type,
        line_index,
        note_line_layer,
        color_type,
        cut_direction,
    )
}

fn clamp_i32(value: i32, min_value: i32, max_value: i32) -> i32 {
    value.max(min_value).min(max_value)
}

fn clamp_f32(value: f32, min_value: f32, max_value: f32) -> f32 {
    value.max(min_value).min(max_value)
}

fn round_half_up(value: f32) -> u32 {
    let floor = value.floor();
    let frac = value - floor;
    if frac < 0.5 {
        floor.max(0.0) as u32
    } else {
        (floor + 1.0).max(0.0) as u32
    }
}

pub fn calc_note_score(cut: &Cut, scoring_type: u32) -> (u32, u32, u32) {
    if !cut.direction_ok || !cut.saber_type_ok || !cut.speed_ok {
        return (0, 0, 0);
    }

    let before_cut_raw_score = if scoring_type == NOTE_SCORE_TYPE_BURSTSLIDERELEMENT {
        0
    } else if scoring_type == NOTE_SCORE_TYPE_SLIDERTAIL {
        70
    } else {
        let score = round_half_up(70.0 * cut.before_cut_rating);
        clamp_i32(score as i32, 0, 70) as u32
    };

    let after_cut_raw_score = if scoring_type == NOTE_SCORE_TYPE_BURSTSLIDERELEMENT {
        0
    } else if scoring_type == NOTE_SCORE_TYPE_BURSTSLIDERHEAD {
        0
    } else if scoring_type == NOTE_SCORE_TYPE_SLIDERHEAD {
        30
    } else {
        let score = round_half_up(30.0 * cut.after_cut_rating);
        clamp_i32(score as i32, 0, 30) as u32
    };

    let cut_distance_raw_score = if scoring_type == NOTE_SCORE_TYPE_BURSTSLIDERELEMENT {
        20
    } else {
        let normalized = 1.0 - clamp_f32(cut.cut_distance_to_center / 0.3, 0.0, 1.0);
        round_half_up(15.0 * normalized)
    };

    (
        before_cut_raw_score,
        after_cut_raw_score,
        cut_distance_raw_score,
    )
}

pub mod base64_bytes {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        STANDARD
            .decode(encoded.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}
