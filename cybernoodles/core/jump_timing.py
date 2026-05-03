INITIAL_HALF_JUMP_BEATS = 4.0
MIN_HALF_JUMP_BEATS = 1.0
MAX_HALF_JUMP_DISTANCE_METERS = 18.0


def compute_spawn_ahead_beats(bpm, note_jump_speed, note_jump_offset=0.0):
    safe_bpm = max(float(bpm), 1e-6)
    safe_njs = max(float(note_jump_speed), 0.0)
    half_jump_beats = INITIAL_HALF_JUMP_BEATS
    seconds_per_beat = 60.0 / safe_bpm

    while (
        half_jump_beats > MIN_HALF_JUMP_BEATS
        and (safe_njs * seconds_per_beat * half_jump_beats) > MAX_HALF_JUMP_DISTANCE_METERS
    ):
        half_jump_beats *= 0.5

    return max(MIN_HALF_JUMP_BEATS, half_jump_beats + float(note_jump_offset))
