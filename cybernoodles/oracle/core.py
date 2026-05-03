"""Minimal Beat Saber replay scoring diagnostic derived from SimSaber.

This module keeps only the replay-score validation path that we actually use:
note motion, saber motion buffering, cut-event scoring, and combo weighting.
It is a diagnostic/reference check, not live training truth, and deliberately
does not try to be a full RL environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, asin, atan2, cos, pi, sin, sqrt
from typing import Dict, Iterable, List, Optional, Tuple

from bsor.Bsor import (
    NOTE_EVENT_BAD,
    NOTE_EVENT_BOMB,
    NOTE_EVENT_GOOD,
    NOTE_EVENT_MISS,
    NOTE_SCORE_TYPE_BURSTSLIDERELEMENT,
    NOTE_SCORE_TYPE_BURSTSLIDERHEAD,
    NOTE_SCORE_TYPE_IGNORE,
    NOTE_SCORE_TYPE_NORMAL_1,
    NOTE_SCORE_TYPE_NORMAL_2,
    NOTE_SCORE_TYPE_NOSCORE,
    NOTE_SCORE_TYPE_SLIDERHEAD,
    NOTE_SCORE_TYPE_SLIDERTAIL,
    Bsor,
    Frame,
    VRObject,
)

DEG_TO_RAD = 0.0174532924
RAD_TO_DEG = 57.29578
BUFFER_SIZE = 500
TWO_PI = 2.0 * pi
NON_SCORING_NOTE_EVENTS = {
    int(NOTE_EVENT_BAD),
    int(NOTE_EVENT_MISS),
    int(NOTE_EVENT_BOMB),
}
NOTE_ID_SCORING_TYPE_FACTOR = 10000
NOTE_ID_PHYSICAL_MODULUS = 10000
SCORABLE_NOTE_SCORE_TYPES = {
    int(NOTE_SCORE_TYPE_NORMAL_1),
    int(NOTE_SCORE_TYPE_NORMAL_2),
    int(NOTE_SCORE_TYPE_SLIDERHEAD),
    int(NOTE_SCORE_TYPE_SLIDERTAIL),
    int(NOTE_SCORE_TYPE_BURSTSLIDERHEAD),
    int(NOTE_SCORE_TYPE_BURSTSLIDERELEMENT),
}
NON_SCORING_NOTE_SCORE_TYPES = {
    int(NOTE_SCORE_TYPE_IGNORE),
    int(NOTE_SCORE_TYPE_NOSCORE),
}


class Quaternion:
    def __init__(self, x: float, y: float, z: float, w: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z,
                self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x,
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            )
        return other * self

    def __rmul__(self, scalar: float) -> "Quaternion":
        return Quaternion(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)

    def __truediv__(self, scalar: float) -> "Quaternion":
        return Quaternion(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)

    def conjugate(self) -> "Quaternion":
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def dot(self, other: "Quaternion") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    @staticmethod
    def from_rotation_matrix(matrix) -> "Quaternion":
        w = sqrt(max(1e-8, 1.0 + matrix[0][0] + matrix[1][1] + matrix[2][2])) * 0.5
        x = (matrix[2][1] - matrix[1][2]) / max(1e-8, 4.0 * w)
        y = (matrix[0][2] - matrix[2][0]) / max(1e-8, 4.0 * w)
        z = (matrix[1][0] - matrix[0][1]) / max(1e-8, 4.0 * w)
        return Quaternion(x, y, z, w)

    @staticmethod
    def from_forward_and_up(forward: "Vector3", up: "Vector3") -> "Quaternion":
        x_axis = up.cross(forward).normal()
        y_axis = forward.cross(x_axis).normal()
        z_axis = forward.normal()
        return Quaternion.from_rotation_matrix(
            (
                (x_axis.x, y_axis.x, z_axis.x),
                (x_axis.y, y_axis.y, z_axis.y),
                (x_axis.z, y_axis.z, z_axis.z),
            )
        )

    @staticmethod
    def slerp(q0: "Quaternion", q1: "Quaternion", t: float) -> "Quaternion":
        t = min(1.0, max(0.0, float(t)))
        dot = max(-1.0, min(1.0, q0.dot(q1)))
        theta = acos(dot)
        if theta <= 1e-8:
            return q0
        out = (sin((1.0 - t) * theta) / sin(theta)) * q0 + (sin(t * theta) / sin(theta)) * q1
        norm = sqrt(max(1e-8, out.dot(out)))
        return out / norm

    @staticmethod
    def lerp(q0: "Quaternion", q1: "Quaternion", t: float) -> "Quaternion":
        t = min(1.0, max(0.0, float(t)))
        out = q0 + (q1 - q0) * t
        norm = sqrt(max(1e-8, out.dot(out)))
        return out / norm

    @staticmethod
    def from_euler(yaw: float, pitch: float, roll: float) -> "Quaternion":
        yaw *= DEG_TO_RAD
        pitch *= DEG_TO_RAD
        roll *= DEG_TO_RAD

        yaw_over_2 = yaw * 0.5
        pitch_over_2 = pitch * 0.5
        roll_over_2 = roll * 0.5
        cos_yaw = cos(yaw_over_2)
        sin_yaw = sin(yaw_over_2)
        cos_pitch = cos(pitch_over_2)
        sin_pitch = sin(pitch_over_2)
        cos_roll = cos(roll_over_2)
        sin_roll = sin(roll_over_2)

        return Quaternion(
            sin_yaw * cos_pitch * cos_roll + cos_yaw * sin_pitch * sin_roll,
            cos_yaw * sin_pitch * cos_roll - sin_yaw * cos_pitch * sin_roll,
            cos_yaw * cos_pitch * sin_roll - sin_yaw * sin_pitch * cos_roll,
            cos_yaw * cos_pitch * cos_roll + sin_yaw * sin_pitch * sin_roll,
        )

    def to_euler(self) -> "Vector3":
        unit = self.dot(self)
        test = self.x * self.w - self.y * self.z
        vec = Vector3(0.0, 0.0, 0.0)

        if test > 0.4995 * unit:
            vec.x = pi / 2.0
            vec.y = 2.0 * atan2(self.y, self.x)
            vec.z = 0.0
        elif test < -0.4995 * unit:
            vec.x = -pi / 2.0
            vec.y = -2.0 * atan2(self.y, self.x)
            vec.z = 0.0
        else:
            vec.x = asin(2.0 * (self.w * self.x - self.y * self.z))
            vec.y = atan2(2.0 * self.w * self.y + 2.0 * self.z * self.x, 1.0 - 2.0 * (self.x * self.x + self.y * self.y))
            vec.z = atan2(2.0 * self.w * self.z + 2.0 * self.x * self.y, 1.0 - 2.0 * (self.z * self.z + self.x * self.x))

        vec *= RAD_TO_DEG
        vec.x %= 360.0
        vec.y %= 360.0
        vec.z %= 360.0
        return vec


class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def sqr_mag(self) -> float:
        return self.dot(self)

    def mag(self) -> float:
        return sqrt(self.sqr_mag())

    def normal(self) -> "Vector3":
        magnitude = self.mag()
        if magnitude <= 1e-8:
            return Vector3(0.0, 0.0, 0.0)
        return self / magnitude

    def angle(self, other: "Vector3") -> float:
        denom = sqrt(max(1e-8, self.sqr_mag() * other.sqr_mag()))
        cosine = self.dot(other) / denom
        cosine = max(-1.0, min(1.0, cosine))
        return acos(cosine) * RAD_TO_DEG

    def rotate(self, quat: Quaternion) -> "Vector3":
        q_form = quat * Quaternion(self.x, self.y, self.z, 0.0) * quat.conjugate()
        return Vector3(q_form.x, q_form.y, q_form.z)

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vector3":
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    @staticmethod
    def distance(a: "Vector3", b: "Vector3") -> float:
        return (a - b).mag()


@dataclass
class Orientation:
    position: Vector3
    rotation: Quaternion


@dataclass
class Plane:
    normal: Vector3
    center: Vector3

    def side(self, point: Vector3) -> int:
        dot = self.normal.dot(point - self.center)
        if dot > 0.0:
            return 1
        if dot < 0.0:
            return -1
        return 0

    def dist_to_point(self, point: Vector3) -> float:
        return abs(self.line_trace(point, self.normal)[0])

    def line_trace(self, point: Vector3, direction: Vector3) -> Tuple[Optional[float], Optional[Vector3]]:
        if self.side(point) == 0:
            return 0.0, point
        denom = self.normal.dot(direction)
        if abs(denom) <= 1e-8:
            return None, None
        dist = self.normal.dot(self.center - point) / denom
        return dist, point + dist * direction

    def ray_trace(self, point: Vector3, direction: Vector3) -> Tuple[Optional[float], Optional[Vector3]]:
        dist, out = self.line_trace(point, direction)
        if dist is None or dist < 0.0:
            return None, None
        return dist, out


@dataclass
class OracleObstacle:
    width: int
    line_index: int
    time: float
    obstacle_type: int
    duration: float


@dataclass
class OracleNote:
    time: float
    note_type: int
    line_index: int
    line_layer: int
    cut_direction: int


@dataclass
class OracleBeatMap:
    difficulty: str
    note_jump_movement_speed: float
    note_jump_start_beat_offset: float
    notes: List[OracleNote] = field(default_factory=list)
    obstacles: List[OracleObstacle] = field(default_factory=list)


@dataclass
class OracleMap:
    beats_per_minute: float
    beatmaps: Dict[str, Dict[str, OracleBeatMap]]


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _note_color_type(note: OracleNote) -> int:
    note_type = _coerce_int(note.note_type)
    return 3 if note_type == 3 else max(0, note_type)


def _note_physical_ids(note: OracleNote) -> Tuple[int, ...]:
    line_index = _coerce_int(note.line_index)
    line_layer = _coerce_int(note.line_layer)
    color_type = _note_color_type(note)
    cut_directions = [_coerce_int(note.cut_direction)]
    if _coerce_int(note.note_type) == 3 and 9 not in cut_directions:
        cut_directions.append(9)
    ids = {
        line_index * 1000 + line_layer * 100 + color_type * 10 + cut_direction
        for cut_direction in cut_directions
    }
    return tuple(sorted(ids))


def _note_id_physical_part(note_id: int) -> int:
    note_id = _coerce_int(note_id)
    if abs(note_id) >= NOTE_ID_SCORING_TYPE_FACTOR:
        return note_id % NOTE_ID_PHYSICAL_MODULUS
    return note_id


def _note_id_scoring_type(note_id: int, default: int = NOTE_SCORE_TYPE_NORMAL_2) -> int:
    note_id = _coerce_int(note_id, default)
    if abs(note_id) >= NOTE_ID_SCORING_TYPE_FACTOR:
        return note_id // NOTE_ID_SCORING_TYPE_FACTOR
    return int(default)


def _full_note_ids_for_physical_id(physical_id: int, scoring_types: Iterable[int]) -> Tuple[int, ...]:
    ids = {int(physical_id)}
    for scoring_type in scoring_types:
        ids.add(int(scoring_type) * NOTE_ID_SCORING_TYPE_FACTOR + int(physical_id))
    return tuple(sorted(ids))


@dataclass
class SaberMovementData:
    hilt_pos: Vector3
    tip_pos: Vector3
    cut_plane_normal: Optional[Vector3]
    time: float
    segment_angle: float

    @classmethod
    def from_points(
        cls,
        hilt: Vector3,
        tip: Vector3,
        prev_hilt: Optional[Vector3],
        prev_tip: Optional[Vector3],
        time: float,
    ) -> "SaberMovementData":
        if prev_hilt is None or prev_tip is None:
            return cls(hilt, tip, None, float(time), 0.0)
        cut_plane_normal = (tip - hilt).cross((prev_hilt + prev_tip) / 2.0 - hilt).normal()
        segment_angle = (tip - hilt).angle(prev_tip - prev_hilt)
        return cls(hilt, tip, cut_plane_normal, float(time), float(segment_angle))


class SaberMovementBuffer:
    def __init__(self):
        self.data: List[Optional[SaberMovementData]] = [None] * BUFFER_SIZE
        self.next_add_index = 0

    def get_curr(self) -> Optional[SaberMovementData]:
        return self.data[(self.next_add_index - 1) % BUFFER_SIZE]

    def get_prev(self) -> Optional[SaberMovementData]:
        return self.data[(self.next_add_index - 2) % BUFFER_SIZE]

    def add_saber_data(self, hand_object: VRObject, time: float):
        new_hilt = Vector3(hand_object.x, hand_object.y, hand_object.z)
        rotation = Quaternion(hand_object.x_rot, hand_object.y_rot, hand_object.z_rot, hand_object.w_rot)
        new_tip = new_hilt + Vector3(0.0, 0.0, 1.0).rotate(rotation)
        curr = self.get_curr()
        if curr is None:
            new_data = SaberMovementData.from_points(new_hilt, new_tip, None, None, time)
        else:
            new_data = SaberMovementData.from_points(new_hilt, new_tip, curr.hilt_pos, curr.tip_pos, time)
        self.data[self.next_add_index] = new_data
        self.next_add_index = (self.next_add_index + 1) % BUFFER_SIZE

    def __iter__(self) -> Iterable[SaberMovementData]:
        rel_index = 0
        while rel_index < BUFFER_SIZE:
            out = self.data[(self.next_add_index - rel_index - 1) % BUFFER_SIZE]
            if out is None:
                return
            rel_index += 1
            yield out

    def calculate_swing_rating(self, override: Optional[float] = None) -> float:
        iterator = iter(self)
        first = next(iterator, None)
        if first is None or first.cut_plane_normal is None:
            return 0.0

        first_normal = first.cut_plane_normal
        first_time = first.time
        prev_time = first_time
        swing_rating = ((first.segment_angle if override is None else override) / 100.0)

        for saber_data in iterator:
            if saber_data.cut_plane_normal is None:
                break
            if first_time - prev_time >= 0.4:
                break

            angle_with_normal = first_normal.angle(saber_data.cut_plane_normal)
            if angle_with_normal >= 90.0:
                break

            prev_time = saber_data.time
            if angle_with_normal < 75.0:
                swing_rating += saber_data.segment_angle / 100.0
            else:
                swing_rating += saber_data.segment_angle * (90.0 - angle_with_normal) / 15.0 / 100.0

            if swing_rating > 1.0:
                return 1.0

        return max(0.0, min(1.0, swing_rating))


class GoodCutEvent:
    def __init__(self, buffer: SaberMovementBuffer, note_orientation: Orientation, cut_point: Optional[Vector3] = None):
        self.note_orientation = note_orientation
        self.buffer = buffer
        self.finished = False

        last_added = buffer.get_curr()
        if last_added is None or last_added.cut_plane_normal is None:
            raise RuntimeError("Cannot score a cut event without valid saber motion history")

        self.cut_plane_normal = last_added.cut_plane_normal
        self.cut_time = last_added.time
        self.before_cut_rating = buffer.calculate_swing_rating()
        self.after_cut_rating = 0.0
        self.note_forward = Vector3(0.0, 0.0, 1.0).rotate(self.note_orientation.rotation)
        self.note_plane = Plane(self.cut_plane_normal.cross(self.note_forward), note_orientation.position)
        self.has_note_plane_been_cut = False

        cut_point = last_added.hilt_pos if cut_point is None else cut_point
        self.cut_plane = Plane(self.cut_plane_normal, cut_point)
        self.acc = self.calculate_acc()

    def update(self):
        curr_data = self.buffer.get_curr()
        prev_data = self.buffer.get_prev()
        if curr_data is None:
            self.finished = True
            return
        if curr_data.time - self.cut_time > 0.4:
            self.finished = True
            return
        if prev_data is None:
            return

        if not self.has_note_plane_been_cut:
            self.note_plane.center = self.note_orientation.position
            self.note_plane.normal = self.cut_plane_normal.cross(self.note_forward)

        if self.note_plane.side(curr_data.tip_pos) != self.note_plane.side(prev_data.tip_pos):
            self.on_intersect_note_plane()
            self.has_note_plane_been_cut = True
        else:
            self.update_after_cut(curr_data)

    def on_intersect_note_plane(self):
        right_before = self.buffer.get_prev()
        right_after = self.buffer.get_curr()
        if right_before is None or right_after is None:
            self.finished = True
            return

        cut_hilt_pos = (right_before.hilt_pos + right_after.hilt_pos) / 2.0
        _, cut_tip_pos = self.note_plane.ray_trace(
            right_before.tip_pos,
            right_after.tip_pos - right_before.tip_pos,
        )
        if cut_tip_pos is None:
            self.finished = True
            return

        self.cut_time = right_after.time
        before_cut_error = (cut_tip_pos - cut_hilt_pos).angle(right_before.tip_pos - right_before.hilt_pos)
        after_cut_error = (cut_tip_pos - cut_hilt_pos).angle(right_after.tip_pos - right_after.hilt_pos)
        self.before_cut_rating = self.buffer.calculate_swing_rating(before_cut_error)
        self.after_cut_rating = after_cut_error / 60.0

    def update_after_cut(self, new_data: SaberMovementData):
        if new_data.cut_plane_normal is None:
            self.finished = True
            return
        angle_with_normal = self.cut_plane_normal.angle(new_data.cut_plane_normal)
        if angle_with_normal >= 90.0:
            self.finished = True
            return

        if angle_with_normal < 75.0:
            self.after_cut_rating += new_data.segment_angle / 60.0
        else:
            self.after_cut_rating += new_data.segment_angle * (90.0 - angle_with_normal) / 15.0 / 60.0

        if self.after_cut_rating > 1.0:
            self.after_cut_rating = 1.0
            self.finished = True

    def calculate_acc(self) -> int:
        dist = self.cut_plane.dist_to_point(self.note_orientation.position)
        acc_percentage = 0.0 if dist > 0.3 else 1.0 - dist / 0.3
        return int(round(acc_percentage * 15.0))

    def get_score(self) -> int:
        pre, post, acc = self.get_score_breakdown()
        return pre + post + acc

    def get_score_breakdown(self) -> Tuple[int, int, int]:
        return (
            int(round(self.before_cut_rating * 70.0)),
            int(round(self.after_cut_rating * 30.0)),
            int(self.acc),
        )


class ComboManager:
    def __init__(self):
        self.meter = 0

    def multiplier(self) -> int:
        if self.meter == 0:
            return 1
        if self.meter < 5:
            return 2
        if self.meter < 13:
            return 4
        return 8

    def __mul__(self, value: int) -> int:
        return value * self.multiplier()

    def __rmul__(self, value: int) -> int:
        return value * self.multiplier()

    def increment(self):
        self.meter += 1


class ScoreManager:
    def __init__(self):
        self.combo = ComboManager()
        self.active_cut_events: List[GoodCutEvent] = []
        self.score = 0
        self.raw_breakdown = [0, 0, 0]
        self.scores: List[Tuple[int, int, int]] = []

    def register_cut_event(self, cut_event: GoodCutEvent):
        self.active_cut_events.append(cut_event)

    def update(self):
        pending: List[GoodCutEvent] = []
        for cut_event in self.active_cut_events:
            cut_event.update()
            if cut_event.finished:
                breakdown = cut_event.get_score_breakdown()
                self.combo.increment()
                self.score += cut_event.get_score() * self.combo
                self.raw_breakdown[0] += breakdown[0]
                self.raw_breakdown[1] += breakdown[1]
                self.raw_breakdown[2] += breakdown[2]
                self.scores.append(breakdown)
            else:
                pending.append(cut_event)
        self.active_cut_events = pending

    def finish(self):
        for cut_event in self.active_cut_events:
            breakdown = cut_event.get_score_breakdown()
            self.combo.increment()
            self.score += cut_event.get_score() * self.combo
            self.raw_breakdown[0] += breakdown[0]
            self.raw_breakdown[1] += breakdown[1]
            self.raw_breakdown[2] += breakdown[2]
            self.scores.append(breakdown)
        self.active_cut_events = []

    def get_score(self) -> int:
        return int(self.score)


def lerp_unclamped(a, b, t):
    return a + (b - a) * t


def lerp(a, b, t):
    return lerp_unclamped(a, b, min(1.0, max(0.0, float(t))))


def quadratic_in_out(t):
    if t < 0.5:
        return 2.0 * t * t
    return (4.0 - 2.0 * t) * t - 1.0


def head_offset_z(note_inverse_world_rotation: Quaternion, head_pseudo_local_pos: Vector3) -> float:
    return head_pseudo_local_pos.rotate(note_inverse_world_rotation).z


def get_z_pos(start: float, end: float, head_offset: float, t: float) -> float:
    return lerp_unclamped(start + head_offset * min(1.0, t * 2.0), end + head_offset, t)


def move_towards_head(start: float, end: float, note_inverse_world_rotation: Quaternion, t: float, head_pseudo_local_pos: Vector3) -> float:
    return get_z_pos(start, end, head_offset_z(note_inverse_world_rotation, head_pseudo_local_pos), t)


def look_rotation(forwards: Vector3, up: Vector3) -> Quaternion:
    return Quaternion.from_forward_and_up(forwards, up)


class NoteData:
    class GameplayType:
        NORMAL = 0

    class ColorType:
        NONE = -1
        COLOR_A = 0
        COLOR_B = 1

    class CutDirection:
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3
        UP_LEFT = 4
        UP_RIGHT = 5
        DOWN_LEFT = 6
        DOWN_RIGHT = 7
        ANY = 8
        NONE = 9

    def __init__(self, oracle_map: OracleMap, note: OracleNote):
        self.time = note.time * 60.0 / oracle_map.beats_per_minute
        self.line_index = note.line_index
        self.flip_line_index = note.line_index
        self.flip_y_side = 0
        self.cut_direction_angle_offset = 0
        self.line_layer = note.line_layer
        self.before_line_layer = 0
        self.note_type = note.note_type
        self.cut_direction = note.cut_direction
        self.gameplay_type = NoteData.GameplayType.NORMAL
        if note.note_type == 0:
            self.color_type = NoteData.ColorType.COLOR_A
        elif note.note_type == 1:
            self.color_type = NoteData.ColorType.COLOR_B
        else:
            self.color_type = NoteData.ColorType.NONE


class MovementData:
    move_speed = 200
    move_duration = 1.0
    center_pos = Vector3(0.0, 0.0, 0.65)

    def __init__(self, oracle_map: OracleMap, note_data: NoteData, replay: Bsor, beatmap: OracleBeatMap):
        self.note_lines_count = 4
        start_njs = beatmap.note_jump_movement_speed
        self.jump_duration = replay.info.jumpDistance / start_njs
        self.right_vec = Vector3(1.0, 0.0, 0.0)
        forward_vec = Vector3(0.0, 0.0, 1.0)
        self.move_distance = self.move_duration * self.move_duration
        self.jump_distance = start_njs * self.jump_duration
        self.move_end_pos = self.center_pos + forward_vec * (self.jump_distance * 0.5)
        self.jump_end_pos = self.center_pos - forward_vec * (self.jump_distance * 0.5)
        self.move_start_pos = self.center_pos + forward_vec * (self.move_distance + self.jump_distance * 0.5)
        self.spawn_ahead_time = self.move_duration + self.jump_duration * 0.5
        self.njs = start_njs
        self.jump_distance_replay = replay.info.jumpDistance
        self.jump_offset_y = self.get_y_offset_from_height(replay.info.height)
        self.end_rotation = self.get_rotation_angle(note_data.cut_direction) + note_data.cut_direction_angle_offset

        note_offset_1 = self.get_note_offset(note_data.line_index, note_data.before_line_layer)
        self.jump_end_pos += note_offset_1
        if note_data.color_type != NoteData.ColorType.NONE:
            note_offset_2 = self.get_note_offset(note_data.flip_line_index, note_data.before_line_layer)
            self.move_start_pos += note_offset_2
            self.move_end_pos += note_offset_2
        else:
            self.move_start_pos += note_offset_1
            self.move_end_pos += note_offset_1

        self.jump_gravity = self.get_gravity(note_data.line_layer, note_data.before_line_layer)
        self.z_offset = 0.25
        self.move_start_pos.z += self.z_offset
        self.move_end_pos.z += self.z_offset
        self.jump_end_pos.z += self.z_offset

    @staticmethod
    def clamp(num: float, min_value: float, max_value: float) -> float:
        return max(min(num, max_value), min_value)

    def get_y_offset_from_height(self, player_height: float) -> float:
        return self.clamp((player_height - 1.7999999523162842) * 0.5, -0.2, 0.6)

    def get_note_offset(self, line_index: int, before_note_line_layer: int) -> Vector3:
        return self.right_vec * ((-(self.note_lines_count - 1) * 0.5 + line_index) * 0.6) + Vector3(
            0.0,
            self.get_y_pos_from_layer(before_note_line_layer),
            0.0,
        )

    @staticmethod
    def get_y_pos_from_layer(layer: int) -> float:
        if layer == 0:
            return 0.25
        if layer == 1:
            return 0.85
        return 1.45

    def highest_jump_pos_y_for_line_layer(self, layer: int) -> float:
        if layer == 0:
            return 0.85 + self.jump_offset_y
        if layer == 1:
            return 1.4 + self.jump_offset_y
        return 1.9 + self.jump_offset_y

    def get_gravity(self, line_layer: int, before_jump_line_layer: int) -> float:
        half_jump = self.jump_distance_replay / self.njs * 0.5
        highest_pos = self.highest_jump_pos_y_for_line_layer(line_layer)
        layer_height = self.get_y_pos_from_layer(before_jump_line_layer)
        return 2.0 * (highest_pos - layer_height) / max(1e-6, half_jump * half_jump)

    @staticmethod
    def get_rotation_angle(cut_direction: int) -> float:
        mapping = {
            NoteData.CutDirection.UP: -180.0,
            NoteData.CutDirection.DOWN: 0.0,
            NoteData.CutDirection.LEFT: -90.0,
            NoteData.CutDirection.RIGHT: 90.0,
            NoteData.CutDirection.UP_LEFT: -135.0,
            NoteData.CutDirection.UP_RIGHT: 135.0,
            NoteData.CutDirection.DOWN_LEFT: -45.0,
            NoteData.CutDirection.DOWN_RIGHT: 45.0,
        }
        return mapping.get(cut_direction, 0.0)


RANDOM_ROTATIONS = [
    Vector3(-0.9543871, -0.1183784, 0.2741019),
    Vector3(0.7680854, -0.08805521, 0.6342642),
    Vector3(-0.6780157, 0.306681, -0.6680131),
    Vector3(0.1255014, 0.9398643, 0.3176546),
    Vector3(0.365105, -0.3664974, -0.8557909),
    Vector3(-0.8790653, -0.06244748, -0.4725934),
    Vector3(0.01886305, -0.8065798, 0.5908241),
    Vector3(-0.1455435, 0.8901445, 0.4318099),
    Vector3(0.07651193, 0.9474725, -0.3105508),
    Vector3(0.1306983, -0.2508438, -0.9591639),
]


def create_note_orientation_updater(oracle_map: OracleMap, note: OracleNote, replay: Bsor, beatmap: OracleBeatMap):
    note_data = NoteData(oracle_map, note)
    movement_data = MovementData(oracle_map, note_data, replay, beatmap)
    movement_start_time = note_data.time - movement_data.move_duration - movement_data.jump_duration / 2.0
    jump_start_time = note_data.time - movement_data.jump_duration / 2.0
    move_duration = movement_data.move_duration
    jump_duration = movement_data.jump_duration
    floor_movement_start_pos = movement_data.move_start_pos
    floor_movement_end_pos = movement_data.move_end_pos
    jump_end_pos = movement_data.jump_end_pos
    gravity = movement_data.jump_gravity
    start_vertical_velocity = gravity * movement_data.jump_duration / 2.0
    y_avoidance = note_data.flip_y_side * 0.15 if note_data.flip_y_side <= 0 else note_data.flip_y_side * 0.45

    end_rotation = Quaternion.from_euler(0.0, 0.0, movement_data.end_rotation)
    euler_angles = end_rotation.to_euler()
    if note_data.gameplay_type == NoteData.GameplayType.NORMAL:
        index = abs(round(note_data.time * 10.0 + jump_end_pos.x * 2.0 + jump_end_pos.y * 2.0)) % len(RANDOM_ROTATIONS)
        euler_angles += RANDOM_ROTATIONS[index] * 20.0
    middle_rotation = Quaternion.from_euler(euler_angles.x, euler_angles.y, euler_angles.z)

    start_rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
    rotate_towards_player = note_data.gameplay_type == NoteData.GameplayType.NORMAL
    world_rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
    inverse_world_rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
    world_to_player_rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
    end_distance_offset = 500.0

    def update(frame: Frame, obj: Orientation):
        time = frame.time
        relative_time = time - movement_start_time

        if relative_time < move_duration:
            obj.position = lerp(floor_movement_start_pos, floor_movement_end_pos, relative_time / move_duration).rotate(world_rotation)
            obj.rotation = start_rotation
            return

        relative_time = time - jump_start_time
        start_pos = floor_movement_end_pos
        end_pos = jump_end_pos
        percentage_of_jump = relative_time / max(1e-6, jump_duration)

        local_pos = Vector3(0.0, 0.0, 0.0)
        if start_pos.x == end_pos.x:
            local_pos.x = start_pos.x
        elif percentage_of_jump >= 0.25:
            local_pos.x = end_pos.x
        else:
            local_pos.x = lerp_unclamped(start_pos.x, end_pos.x, quadratic_in_out(percentage_of_jump * 4.0))

        local_pos.y = start_pos.y + start_vertical_velocity * relative_time - gravity * relative_time * relative_time * 0.5
        head_pseudo_local_pos = Vector3(frame.head.x, frame.head.y, frame.head.z)
        local_pos.z = move_towards_head(start_pos.z, end_pos.z, inverse_world_rotation, percentage_of_jump, head_pseudo_local_pos)

        if y_avoidance != 0.0 and percentage_of_jump < 0.25:
            local_pos.y += (0.5 - cos(percentage_of_jump * 8.0 * pi) * 0.5) * y_avoidance

        if percentage_of_jump < 0.5:
            if percentage_of_jump >= 0.125:
                a = Quaternion.slerp(middle_rotation, end_rotation, sin((percentage_of_jump - 0.125) * pi * 2.0))
            else:
                a = Quaternion.slerp(start_rotation, middle_rotation, sin(percentage_of_jump * pi * 4.0))

            if rotate_towards_player:
                head_pseudo_location = Vector3(frame.head.x, frame.head.y, frame.head.z)
                head_pseudo_location.y = lerp(head_pseudo_location.y, local_pos.y, 0.8)
                normalized = (local_pos - head_pseudo_location.rotate(inverse_world_rotation)).normal()
                rotated_object_up = Vector3(0.0, 1.0, 0.0).rotate(obj.rotation)
                vector_up = rotated_object_up.rotate(world_to_player_rotation)
                b = look_rotation(normalized, vector_up.rotate(inverse_world_rotation))
                obj.rotation = Quaternion.lerp(a, b, percentage_of_jump * 2.0)
            else:
                obj.rotation = a

        if percentage_of_jump >= 0.75:
            num2 = (percentage_of_jump - 0.75) / 0.25
            local_pos.z -= lerp_unclamped(0.0, end_distance_offset, num2 * num2 * num2)

        obj.position = local_pos.rotate(world_rotation)

    return update


class NoteObject:
    def __init__(self, oracle_map: OracleMap, beatmap: OracleBeatMap, note: OracleNote, replay: Bsor, manager: "NoteManager"):
        self.physical_ids = _note_physical_ids(note)
        self.id = int(NOTE_SCORE_TYPE_NORMAL_2) * NOTE_ID_SCORING_TYPE_FACTOR + self.physical_ids[0]
        scoring_types = SCORABLE_NOTE_SCORE_TYPES
        if _coerce_int(note.note_type) == 3:
            scoring_types = NON_SCORING_NOTE_SCORE_TYPES | {int(NOTE_SCORE_TYPE_NORMAL_2)}
        self.note_ids = {
            note_id
            for physical_id in self.physical_ids
            for note_id in _full_note_ids_for_physical_id(physical_id, scoring_types)
        }
        self.updater = create_note_orientation_updater(oracle_map, note, replay, beatmap)
        self.orientation = Orientation(Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))
        self.manager = manager

    def update(self, frame: Frame):
        self.updater(frame, self.orientation)

    def matches_note_id(self, note_id: int) -> bool:
        note_id = _coerce_int(note_id)
        return note_id in self.note_ids or _note_id_physical_part(note_id) in self.physical_ids

    def handle_cut(self):
        if self in self.manager.active:
            self.manager.active.remove(self)


class NoteManager:
    def __init__(self, oracle_map: OracleMap, replay: Bsor):
        mode = str(replay.info.mode)
        difficulty = str(replay.info.difficulty)
        self.beatmap = oracle_map.beatmaps[mode][difficulty]
        self.notes = list(reversed(self.beatmap.notes))
        self.oracle_map = oracle_map
        self.replay = replay
        self.spawn_ahead_time = 1.0 + self.replay.info.jumpDistance / self.beatmap.note_jump_movement_speed * 0.5
        self.active: List[NoteObject] = []

    def update(self, frame: Frame):
        while self.notes and frame.time >= self.get_spawn_time(self.notes[-1]):
            self.active.append(NoteObject(self.oracle_map, self.beatmap, self.notes.pop(), self.replay, self))
        for note_object in list(self.active):
            note_object.update(frame)

    def get_spawn_time(self, note: OracleNote) -> float:
        return note.time * 60.0 / self.oracle_map.beats_per_minute - self.spawn_ahead_time

    def get_active_note_by_id(self, note_id: int) -> Optional[NoteObject]:
        for note_object in self.active:
            if note_object.matches_note_id(note_id):
                return note_object
        return None

    def get_active_note_for_event(self, event) -> Optional[NoteObject]:
        note_id = getattr(event, "note_id", None)
        if note_id is not None:
            note_object = self.get_active_note_by_id(note_id)
            if note_object is not None:
                return note_object

        line_index = getattr(event, "lineIndex", None)
        line_layer = getattr(event, "noteLineLayer", getattr(event, "lineLayer", None))
        color_type = getattr(event, "colorType", None)
        cut_direction = getattr(event, "cutDirection", None)
        if None in (line_index, line_layer, color_type, cut_direction):
            return None

        physical_id = (
            _coerce_int(line_index) * 1000
            + _coerce_int(line_layer) * 100
            + _coerce_int(color_type) * 10
            + _coerce_int(cut_direction)
        )
        for note_object in self.active:
            if physical_id in note_object.physical_ids:
                return note_object
        return None


@dataclass
class OracleScoreResult:
    score: int
    raw_breakdown: Tuple[int, int, int]
    cut_breakdowns: List[Tuple[int, int, int]]
    score_model: str = "oracle_reference_diagnostic"


def _note_event_type(event) -> int:
    try:
        return int(getattr(event, "event_type", NOTE_EVENT_GOOD))
    except (TypeError, ValueError):
        return int(NOTE_EVENT_GOOD)


def _note_event_scoring_type(event) -> int:
    note_id = getattr(event, "note_id", 0)
    note_id_scoring_type = _note_id_scoring_type(note_id, NOTE_SCORE_TYPE_NORMAL_2)
    if abs(_coerce_int(note_id)) >= NOTE_ID_SCORING_TYPE_FACTOR and note_id_scoring_type != int(NOTE_SCORE_TYPE_NORMAL_1):
        return note_id_scoring_type

    explicit = getattr(event, "scoringType", None)
    if explicit is not None:
        return _coerce_int(explicit, NOTE_SCORE_TYPE_NORMAL_2)
    return note_id_scoring_type


def _is_scoring_note_event(event_type: int, scoring_type: int) -> bool:
    if _coerce_int(event_type, NOTE_EVENT_GOOD) != int(NOTE_EVENT_GOOD):
        return False
    return _coerce_int(scoring_type, NOTE_SCORE_TYPE_NORMAL_2) in SCORABLE_NOTE_SCORE_TYPES


def calculate_score_assuming_valid_times(oracle_map: OracleMap, replay: Bsor, verbose: bool = False) -> OracleScoreResult:
    left_hand_buffer = SaberMovementBuffer()
    right_hand_buffer = SaberMovementBuffer()
    note_manager = NoteManager(oracle_map, replay)
    score_manager = ScoreManager()
    note_events = list(reversed(replay.notes))

    frame_count = 0
    for frame in replay.frames[1:]:
        frame_count += 1
        if verbose and frame_count % 1000 == 0:
            print(f"Processed frame {frame_count}")

        left_hand_buffer.add_saber_data(frame.left_hand, frame.time)
        right_hand_buffer.add_saber_data(frame.right_hand, frame.time)

        while note_events and note_events[-1].event_time < frame.time:
            event = note_events.pop()
            note_object = note_manager.get_active_note_for_event(event)
            if note_object is None:
                continue

            note_object.handle_cut()
            event_type = _note_event_type(event)
            scoring_type = _note_event_scoring_type(event)
            if event_type in NON_SCORING_NOTE_EVENTS:
                continue
            if not _is_scoring_note_event(event_type, scoring_type):
                continue

            cut = getattr(event, "cut", None)
            if cut is None:
                continue

            cut_point = Vector3(*cut.cutPoint)
            buffer = left_hand_buffer if int(cut.saberType) == 0 else right_hand_buffer
            score_manager.register_cut_event(GoodCutEvent(buffer, note_object.orientation, cut_point))

        note_manager.update(frame)
        score_manager.update()

    score_manager.finish()
    return OracleScoreResult(
        score=score_manager.get_score(),
        raw_breakdown=tuple(int(x) for x in score_manager.raw_breakdown),
        cut_breakdowns=list(score_manager.scores),
    )
