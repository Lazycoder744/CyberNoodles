from types import SimpleNamespace
from unittest import mock

import pytest
from bsor.Bsor import (
    NOTE_EVENT_BAD,
    NOTE_EVENT_BOMB,
    NOTE_EVENT_GOOD,
    NOTE_EVENT_MISS,
    NOTE_SCORE_TYPE_NOSCORE,
)

from cybernoodles.oracle.core import (
    OracleBeatMap,
    OracleMap,
    OracleNote,
    calculate_score_assuming_valid_times,
)
from cybernoodles.tools.score_replay import (
    compute_max_score,
    compute_oracle_reference_max_score,
    compute_score_maxima,
    compute_standard_max_score,
)


def _note_id(note):
    return 30000 + note.line_index * 1000 + note.line_layer * 100 + note.note_type * 10 + note.cut_direction


def _vr_object(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(
        x=float(x),
        y=float(y),
        z=float(z),
        x_rot=0.0,
        y_rot=0.0,
        z_rot=0.0,
        w_rot=1.0,
    )


def _frame(time, hand_x):
    return SimpleNamespace(
        time=float(time),
        head=_vr_object(0.0, 1.7, 0.0),
        left_hand=_vr_object(hand_x, 1.0, 0.0),
        right_hand=_vr_object(-hand_x, 1.0, 0.0),
    )


def _cut():
    return SimpleNamespace(
        cutPoint=[0.0, 1.0, 0.0],
        saberType=0,
    )


def _oracle_map(note):
    beatmap = OracleBeatMap(
        difficulty="Expert",
        note_jump_movement_speed=18.0,
        note_jump_start_beat_offset=0.0,
        notes=[note],
        obstacles=[],
    )
    return OracleMap(beats_per_minute=120.0, beatmaps={"Standard": {"Expert": beatmap}})


def _replay(note, event_type, cut, note_id=None, scoring_type=None):
    return SimpleNamespace(
        info=SimpleNamespace(
            mode="Standard",
            difficulty="Expert",
            jumpDistance=18.0,
            height=1.7,
        ),
        frames=[
            _frame(0.0, 0.0),
            _frame(0.1, 0.0),
            _frame(0.2, 0.2),
            _frame(0.6, 0.4),
            _frame(1.1, 0.6),
        ],
        notes=[
            SimpleNamespace(
                note_id=_note_id(note) if note_id is None else int(note_id),
                event_time=0.5,
                event_type=event_type,
                cut=cut,
                scoringType=scoring_type,
            )
        ],
    )


class _FakeGoodCutEvent:
    created = []

    def __init__(self, buffer, note_orientation, cut_point):
        self.finished = False
        self.cut_point = cut_point
        self.created.append(self)

    def update(self):
        self.finished = True

    def get_score(self):
        return 115

    def get_score_breakdown(self):
        return (70, 30, 15)


def test_oracle_reference_diagnostic_good_cut_event_scores_once():
    note = OracleNote(time=1.0, note_type=0, line_index=1, line_layer=1, cut_direction=8)
    _FakeGoodCutEvent.created = []

    with mock.patch("cybernoodles.oracle.core.GoodCutEvent", _FakeGoodCutEvent):
        result = calculate_score_assuming_valid_times(_oracle_map(note), _replay(note, NOTE_EVENT_GOOD, _cut()))

    assert result.score == 230
    assert result.raw_breakdown == (70, 30, 15)
    assert result.cut_breakdowns == [(70, 30, 15)]
    assert result.score_model == "oracle_reference_diagnostic"
    assert len(_FakeGoodCutEvent.created) == 1


def test_miss_event_with_no_cut_does_not_crash_or_score():
    note = OracleNote(time=1.0, note_type=0, line_index=1, line_layer=1, cut_direction=8)

    with mock.patch("cybernoodles.oracle.core.GoodCutEvent", side_effect=AssertionError("miss scored")):
        result = calculate_score_assuming_valid_times(_oracle_map(note), _replay(note, NOTE_EVENT_MISS, None))

    assert result.score == 0
    assert result.raw_breakdown == (0, 0, 0)
    assert result.cut_breakdowns == []


def test_bad_cut_event_with_cut_does_not_become_good_cut():
    note = OracleNote(time=1.0, note_type=0, line_index=1, line_layer=1, cut_direction=8)

    with mock.patch("cybernoodles.oracle.core.GoodCutEvent", side_effect=AssertionError("bad cut scored")):
        result = calculate_score_assuming_valid_times(_oracle_map(note), _replay(note, NOTE_EVENT_BAD, _cut()))

    assert result.score == 0
    assert result.raw_breakdown == (0, 0, 0)
    assert result.cut_breakdowns == []


def test_bomb_event_with_no_cut_does_not_crash_or_score():
    note = OracleNote(time=1.0, note_type=3, line_index=1, line_layer=1, cut_direction=8)

    with mock.patch("cybernoodles.oracle.core.GoodCutEvent", side_effect=AssertionError("bomb scored")):
        result = calculate_score_assuming_valid_times(_oracle_map(note), _replay(note, NOTE_EVENT_BOMB, None))

    assert result.score == 0
    assert result.raw_breakdown == (0, 0, 0)
    assert result.cut_breakdowns == []


def test_good_event_with_no_score_scoring_type_does_not_score():
    note = OracleNote(time=1.0, note_type=0, line_index=1, line_layer=1, cut_direction=8)
    physical_note_id = _note_id(note) % 10000

    with mock.patch("cybernoodles.oracle.core.GoodCutEvent", side_effect=AssertionError("noscore event scored")):
        result = calculate_score_assuming_valid_times(
            _oracle_map(note),
            _replay(
                note,
                NOTE_EVENT_GOOD,
                _cut(),
                note_id=physical_note_id,
                scoring_type=NOTE_SCORE_TYPE_NOSCORE,
            ),
        )

    assert result.score == 0
    assert result.raw_breakdown == (0, 0, 0)
    assert result.cut_breakdowns == []
    assert result.score_model == "oracle_reference_diagnostic"


def test_standard_and_oracle_reference_max_scores_are_named_separately():
    map_notes = [
        {"type": 0, "scoreCap": 115.0},
        {"type": 1, "scoreCap": 115.0},
        {"type": 3, "scoreCap": 115.0},
    ]

    assert compute_standard_max_score(map_notes) == 345
    assert compute_oracle_reference_max_score(map_notes) == 460
    assert compute_max_score(map_notes, scoring_model="standard") == 345
    assert compute_max_score(map_notes, scoring_model="oracle_reference") == 460
    assert compute_score_maxima(map_notes) == {
        "standard_max_score": 345,
        "oracle_reference_max_score": 460,
    }


def test_compute_max_score_rejects_unknown_scoring_model_name():
    with pytest.raises(ValueError, match="scoring_model"):
        compute_max_score([{"type": 0, "scoreCap": 115.0}], scoring_model="unknown")
