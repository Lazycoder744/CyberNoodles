import torch
import torch.nn as nn
import numpy as np

# ─── Shared Constants ───────────────────────────────────────────────
# Notes keep the original beat-relative timing for musical context, but also
# expose physical approach timing so the policy can disambiguate maps whose
# notes are the same number of beats away but move at very different speeds.
NOTE_FEATURES      = 10  # [time_beats, x, y, saber_type, cut_dx, cut_dy, score_class, score_cap, time_seconds, z_distance]
NUM_UPCOMING_NOTES = 20
NOTES_DIM          = NOTE_FEATURES * NUM_UPCOMING_NOTES
OBSTACLE_FEATURES  = 6   # [time, x, y, width, height, duration]
NUM_UPCOMING_OBSTACLES = 6
OBSTACLES_DIM      = OBSTACLE_FEATURES * NUM_UPCOMING_OBSTACLES
POSE_DIM           = 21  # Full tracked pose: head(7) + left(7) + right(7)
VELOCITY_DIM       = 21
ACTION_DIM         = 21  # Predict full pose so the policy can learn dodges
STATE_HISTORY_OFFSETS = (0, 2, 4)  # current, short-term, and medium-term motion context
STATE_HISTORY_FRAMES = len(STATE_HISTORY_OFFSETS)
STATE_FRAME_DIM    = POSE_DIM + VELOCITY_DIM
MOTION_DIM         = STATE_HISTORY_FRAMES * STATE_FRAME_DIM
INPUT_DIM          = NOTES_DIM + OBSTACLES_DIM + MOTION_DIM
CURRENT_POSE_START = NOTES_DIM + OBSTACLES_DIM
CURRENT_POSE_END   = CURRENT_POSE_START + POSE_DIM
CURRENT_VELOCITY_START = CURRENT_POSE_END
CURRENT_VELOCITY_END = CURRENT_VELOCITY_START + VELOCITY_DIM
POSE_QUATERNION_SLICES = ((3, 7), (10, 14), (17, 21))

NOTE_TIME_INDEX = 0
NOTE_LANE_INDEX = 1
NOTE_LAYER_INDEX = 2
NOTE_TYPE_INDEX = 3
NOTE_TIME_SECONDS_INDEX = 8
NOTE_Z_DISTANCE_INDEX = 9

# Static head pose (constant — head doesn't affect scoring)
STATIC_HEAD = np.array([0.0, 1.7, 0.0, 0.0, 0.0, 0.0, 1.0])

# ─── Cut Direction Encoding ─────────────────────────────────────────
# Converts integer cutDirection (0-8) into a 2D unit vector (dx, dy)
# that gives the model geometric knowledge of the required swing direction.
CUT_DIR_VECTORS = {
    0: (0.0, 1.0),          # Up
    1: (0.0, -1.0),         # Down
    2: (-1.0, 0.0),         # Left
    3: (1.0, 0.0),          # Right
    4: (-0.7071, 0.7071),   # Up-Left
    5: (0.7071, 0.7071),    # Up-Right
    6: (-0.7071, -0.7071),  # Down-Left
    7: (0.7071, -0.7071),   # Down-Right
    8: (0.0, 0.0),          # Any/Dot
}

def encode_cut_direction(cut_dir):
    """Convert integer cut direction to (dx, dy) unit vector."""
    return CUT_DIR_VECTORS.get(int(cut_dir), (0.0, 0.0))


_pose_identity_quaternions = {}


def get_pose_quaternion_identity(device, dtype):
    if hasattr(device, "type"):
        key = (device.type, device.index if device.index is not None else 0, dtype)
    else:
        key = (str(device), 0, dtype)

    if key not in _pose_identity_quaternions:
        _pose_identity_quaternions[key] = torch.tensor(
            (0.0, 0.0, 0.0, 1.0),
            device=device,
            dtype=dtype,
        )
    return _pose_identity_quaternions[key]


def normalize_pose_quaternions(pose):
    identity = get_pose_quaternion_identity(pose.device, pose.dtype)
    identity_shape = (1,) * max(0, pose.dim() - 1) + (4,)
    cursor = 0
    pieces = []

    for start, end in POSE_QUATERNION_SLICES:
        if cursor < start:
            pieces.append(pose[..., cursor:start])

        quat = pose[..., start:end]
        quat_norm = torch.norm(quat, dim=-1, keepdim=True)
        safe_quat = torch.where(
            quat_norm > 1e-6,
            quat / quat_norm.clamp(min=1e-6),
            identity.view(identity_shape).expand_as(quat),
        )
        pieces.append(safe_quat)
        cursor = end

    if cursor < pose.shape[-1]:
        pieces.append(pose[..., cursor:])

    return torch.cat(pieces, dim=-1)

# ─── State Normalization ────────────────────────────────────────────
# Dict of scratch buffers keyed by shape — each CUDA graph (rollout vs
# PPO) captures a different batch size and needs its own fixed-address
_norm_bufs = {}
_norm_vectors = {}
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0

def get_norm_vector(device):
    # Key on (type, index) tuple — robust against device objects whose str()
    # representation can vary after a CPU round-trip (e.g. 'cuda' vs 'cuda:0').
    key = (device.type, device.index if device.index is not None else 0) \
          if hasattr(device, 'type') else (str(device), 0)
    if key not in _norm_vectors:
        n = NOTES_DIM
        v = torch.ones(INPUT_DIM, device=device)
        
        # Notes: stride 10 —
        # [time_beats, idx, layer, saber_type, dx, dy, score_class, score_cap, time_seconds, z_distance]
        v[0:n:NOTE_FEATURES] = 1.0 / 4.0
        v[1:n:NOTE_FEATURES] = 1.0 / 3.0
        v[2:n:NOTE_FEATURES] = 1.0 / 2.0
        v[3:n:NOTE_FEATURES] = 1.0 / 2.0
        v[4:n:NOTE_FEATURES] = 1.0
        v[5:n:NOTE_FEATURES] = 1.0
        v[6:n:NOTE_FEATURES] = 1.0 / 4.0
        v[7:n:NOTE_FEATURES] = 1.0
        v[8:n:NOTE_FEATURES] = 1.0 / 3.0
        v[9:n:NOTE_FEATURES] = 1.0 / 32.0

        obs_start = NOTES_DIM
        obs_end = obs_start + OBSTACLES_DIM
        # Obstacles: stride 6 — [time, x, y, width, height, duration]
        v[obs_start:obs_end:OBSTACLE_FEATURES] = 1.0 / 4.0
        v[obs_start + 1:obs_end:OBSTACLE_FEATURES] = 1.0 / 3.0
        v[obs_start + 2:obs_end:OBSTACLE_FEATURES] = 1.0 / 3.0
        v[obs_start + 3:obs_end:OBSTACLE_FEATURES] = 1.0 / 4.0
        v[obs_start + 4:obs_end:OBSTACLE_FEATURES] = 1.0 / 5.0
        v[obs_start + 5:obs_end:OBSTACLE_FEATURES] = 1.0 / 8.0
        
        for hist_idx in range(STATE_HISTORY_FRAMES):
            start = NOTES_DIM + OBSTACLES_DIM + hist_idx * STATE_FRAME_DIM
            pose_start = start
            vel_start = start + POSE_DIM

            # Hand pose history slices
            v[pose_start : pose_start + POSE_DIM] = 1.0 / 2.0
            # Hand velocity history slices
            v[vel_start : vel_start + VELOCITY_DIM] = 1.0 / 8.0

        _norm_vectors[key] = v
    return _norm_vectors[key]

def normalize_state(x):
    """Normalize the state tensor to roughly +/- 1.0 range.

    Uses a per-shape cached scratch buffer so the same fixed GPU address
    is reused on every call — required for CUDA graph compatibility.
    """
    vec = get_norm_vector(x.device)
    if torch.is_grad_enabled():
        # Training-time forwards must keep their own normalized tensor alive
        # for autograd; reusing the shared CUDA-graph scratch buffer here can
        # corrupt saved activations when another model forward runs afterward.
        return x * vec

    key = (tuple(x.shape), x.device.type, x.device.index, x.dtype)
    if key not in _norm_bufs:
        _norm_bufs[key] = torch.empty_like(x)
    x_norm = _norm_bufs[key]
    
    # Vectorized normalization in a single kernel call
    # Uses 'out=' to ensure zero allocations during CUDA graph capture.
    torch.mul(x, vec, out=x_norm)

    return x_norm

# ─── Model ──────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Split actor-critic with SEPARATE feature extractors.
    
    Why split? With a shared backbone, the critic's MSE gradients
    (which can be huge early in training) destroy the actor's learned
    features.  Separate networks let each head learn at its own pace.
    
    Architecture is deliberately compact (256-wide) because the input
    is still compact enough that a 256-wide network trains quickly and
    remains practical for PPO updates.

    Action contract: forward() returns the raw Normal policy parameters in
    absolute pose-action space. Callers sample raw_action from (mean, std),
    compute log-probs on that unsanitized sample, and only then sanitize the
    action before passing it to the simulator.
    """
    def __init__(self):
        super(ActorCritic, self).__init__()

        # ── Actor pathway ────────────────────────────────────────────
        self.actor_features = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(256, ACTION_DIM)
        # Learnable log_std. Start low because BC pretraining only learns the
        # mean path; a large initial std turns a good imitation prior into
        # flailing once PPO begins sampling from it.
        self.actor_log_std = nn.Parameter(torch.full((1, ACTION_DIM), -2.3))

        # ── Critic pathway (separate weights) ────────────────────────
        self.critic_features = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        x_normalized = normalize_state(x)
        
        # ── Actor ────────────────────────────────────────────────────
        actor_h = self.actor_features(x_normalized)
        # Residual action space: predict a delta from current hand pose.
        # No tanh/clamp here. The mean is the raw policy Normal location;
        # simulator input clamping belongs at the policy_eval boundary.
        current_pose = x[:, CURRENT_POSE_START:CURRENT_POSE_END]
        mean = normalize_pose_quaternions(current_pose + self.actor_mean(actor_h))
        std = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        
        # ── Critic ───────────────────────────────────────────────────
        critic_h = self.critic_features(x_normalized)
        value = self.critic_head(critic_h)
        
        return mean, std, value


def build_rl_bootstrap_state_dict(model):
    """Export a BC-trained actor with a critic warm-start for PPO handoff."""
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    critic_seed_pairs = (
        ("critic_features.0.weight", "actor_features.0.weight"),
        ("critic_features.0.bias", "actor_features.0.bias"),
        ("critic_features.1.weight", "actor_features.1.weight"),
        ("critic_features.1.bias", "actor_features.1.bias"),
        ("critic_features.3.weight", "actor_features.3.weight"),
        ("critic_features.3.bias", "actor_features.3.bias"),
        ("critic_features.4.weight", "actor_features.4.weight"),
        ("critic_features.4.bias", "actor_features.4.bias"),
    )
    for critic_key, actor_key in critic_seed_pairs:
        if critic_key in state and actor_key in state and state[critic_key].shape == state[actor_key].shape:
            state[critic_key].copy_(state[actor_key])
    if "critic_head.weight" in state:
        state["critic_head.weight"].zero_()
    if "critic_head.bias" in state:
        state["critic_head.bias"].zero_()
    return state

if __name__ == "__main__":
    model = ActorCritic()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    dummy_input = torch.zeros((1, INPUT_DIM))
    mean, std, value = model(dummy_input)
    print(f"Input dim: {INPUT_DIM}, Action dim: {ACTION_DIM}")
    print(f"Mean shape: {mean.shape}, Value shape: {value.shape}")
