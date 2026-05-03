import os, pathlib
import torch._inductor.codecache as _cc

# ── Windows fix: pathlib.rename() fails if destination exists; os.replace() doesn't ──
def _patched_write_atomic(path, content, make_dirs=False, mode="w"):
    if make_dirs:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(str(path) + ".tmp2")
    tmp.write_bytes(content if isinstance(content, bytes) else content.encode())
    os.replace(tmp, path)

_cc.write_atomic = _patched_write_atomic
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time, json
from cybernoodles.core.gpu_simulator import GPUBeatSaberSimulator
from cybernoodles.core.network import ActorCritic
from cybernoodles.data.dataset_builder import get_map_notes
from cybernoodles.data.fetch_data import fetch_random_ranked_maps
from cybernoodles.data.map_analyzer import analyze_all_maps

BC_MODEL_PATH = "bsai_bc_model.pth"


def _remap_state_dict(state_dict, model):
    """Load a state_dict that may come from the OLD model (with Dropout layers).

    Old Sequential indices: 0,1,2,3(Drop),4,5,6,7(Drop),8,9,10,11,12,13
    New Sequential indices: 0,1,2,           3,4,5,       6,7,8, 9,10,11

    If the state_dict keys already match the model, this is a no-op.
    """
    model_keys = set(model.state_dict().keys())
    sd_keys    = set(state_dict.keys())
    if sd_keys == model_keys:
        model.load_state_dict(state_dict)
        return

    # Build mapping: old index → new index (skip Dropout at 3 and 7)
    OLD_TO_NEW = {0:0, 1:1, 2:2,  # first block (before old Dropout 3)
                  4:3, 5:4, 6:5,  # second block (before old Dropout 7)
                  8:6, 9:7, 10:8, # third block
                  11:9, 12:10, 13:11}  # fourth block

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("features."):
            parts = key.split(".")
            old_idx = int(parts[1])
            if old_idx in OLD_TO_NEW:
                parts[1] = str(OLD_TO_NEW[old_idx])
                new_key = ".".join(parts)
            else:
                continue  # skip dropout params (they have none, but just in case)
        remapped[new_key] = value

    model.load_state_dict(remapped, strict=False)


def compute_ppo_loss(states, actions, log_probs_old, returns, advantages, model,
                     clip_param=0.2):
    mean, std, value = model(states)
    dist = Normal(mean, std, validate_args=False)
    log_probs = dist.log_prob(actions).sum(dim=-1)
    ratio = torch.exp(log_probs - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    ppo_loss = (-torch.min(surr1, surr2).mean()
                + 0.5 * nn.MSELoss()(value.squeeze(-1), returns)
                - 0.10 * dist.entropy().mean())

    return ppo_loss


# ─────────────────────────────────────────────────────────────────────────────
# CUDA graph rollout capture
# ─────────────────────────────────────────────────────────────────────────────

def build_cuda_graph(sim, model, device):
    """
    Captures a single rollout step as a CUDA graph.
    """
    N = sim.num_envs

    # Static tensors — addresses captured by the graph
    static_noise  = torch.zeros(N, 14,  device=device)
    static_state  = sim._state_out           # already allocated in sim.reset()
    static_reward = sim._reward_out

    # Warmup
    print("  [CUDA graph] Warming up...", flush=True)
    for _ in range(3):
        s = sim.get_states()
        with torch.no_grad():
            mean, std, value = model(s)
        noise  = torch.randn_like(mean)
        action = mean + std * noise
        sim.step(action)
    torch.cuda.synchronize()

    # Re-initialise sim state with 0 offset for a clean capture context
    sim.reset(None)

    # Allocate output tensors (fixed addresses)
    static_mean   = torch.zeros(N, 14, device=device)
    static_std    = torch.zeros(N, 14, device=device)
    static_value  = torch.zeros(N,  1, device=device)
    static_action = torch.zeros(N, 14, device=device)
    static_logp   = torch.zeros(N,     device=device)

    print("  [CUDA graph] Capturing...", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = sim.get_states()
        with torch.no_grad():
            _mean, _std, _value = model(static_state)
        static_mean.copy_(_mean)
        static_std.copy_(_std)
        static_value.copy_(_value)
        static_action.copy_(static_mean + static_std * static_noise)
        static_logp.copy_(
            Normal(static_mean, static_std, validate_args=False).log_prob(static_action).sum(-1)
        )
        sim.step(static_action)

    torch.cuda.synchronize()
    print("  [CUDA graph] Capture complete.", flush=True)

    return g, static_mean, static_std, static_value, static_action, static_logp, static_state, static_reward, static_noise


# ─────────────────────────────────────────────────────────────────────────────
# CUDA graph PPO update capture
# ─────────────────────────────────────────────────────────────────────────────

def build_ppo_graph(model, optimizer, batch_size, device, clip_param=0.1):
    """
    Captures one PPO gradient step as a CUDA graph.
    """
    B = batch_size
 
    g_states     = torch.zeros(B, 148, device=device)
    g_actions    = torch.zeros(B, 14,  device=device)
    g_logp_old   = torch.zeros(B,      device=device)
    g_returns    = torch.zeros(B,      device=device)
    g_adv        = torch.zeros(B,      device=device)

    print("  [PPO graph] Warming up...", flush=True)
    model.train()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=False)
            loss = compute_ppo_loss(g_states, g_actions, g_logp_old, g_returns, g_adv,
                                    model, clip_param)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None: p.grad.clamp_(-0.5, 0.5)
            optimizer.step()
    torch.cuda.synchronize()

    print("  [PPO graph] Capturing...", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(g, stream=s):
            optimizer.zero_grad(set_to_none=False)
            _loss = compute_ppo_loss(g_states, g_actions, g_logp_old, g_returns, g_adv,
                                     model, clip_param)
            _loss.backward()
            for p in model.parameters():
                if p.grad is not None: p.grad.clamp_(-0.5, 0.5)
            optimizer.step()
    torch.cuda.synchronize()
    print("  [PPO graph] Capture complete.", flush=True)
 
    return g, g_states, g_actions, g_logp_old, g_returns, g_adv


def train_ppo_gpu(epochs=5000, model_path="bsai_rl_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: GPU not available!")
        return

    torch.set_float32_matmul_precision('high')

    print(f"\033[96m{'='*60}")
    print(f"  CyberNoodles 4.0: CUDA Graph PPO Trainer")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\033[0m")

    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, capturable=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    if os.path.exists(model_path):
        _remap_state_dict(torch.load(model_path, map_location=device, weights_only=True), model)
        print(f"\033[92mExisting RL checkpoint loaded.\033[0m")
    
    with torch.no_grad():
        model.actor_log_std.fill_(0.0)
    
    log_std_unfrozen = True

    NUM_ENVS   = 256
    GAMMA      = 0.99
    GAE_LAMBDA = 0.95
    STEPS      = 1200  # 20 seconds of play

    # Curriculum
    curriculum = []
    if os.path.exists('curriculum.json'):
        with open('curriculum.json') as f:
            curriculum = json.load(f)
    curriculum_hashes = [c['hash'] for c in curriculum]

    moving_acc       = 0.0
    current_tier_code = 1
    STATE_PATH = "rl_state.json"
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                s = json.loads(f.read().strip() or '{}')
                moving_acc        = s.get('moving_acc', 0.0)
                current_tier_code = s.get('current_tier_code', 1)
                print(f"\033[94mResumed: Tier {current_tier_code}, Acc {moving_acc:.2f}%\033[0m")
        except Exception:
            pass

    sim = GPUBeatSaberSimulator(NUM_ENVS, device)
    curriculum_hashes = curriculum_hashes[:720]
    
    print(f"\033[94mCaching max {len(curriculum_hashes)} maps into RAM...\033[0m")
    map_cache = {}
    for h in curriculum_hashes:
        notes, bpm = get_map_notes(h)
        if notes:
            map_cache[h] = (notes, bpm)
    curriculum_hashes = list(map_cache.keys())
    print(f"\033[94mCached {len(map_cache)} maps.")

    # Pre-allocate rollout buffers
    states_buf   = torch.zeros(STEPS, NUM_ENVS, 148, device=device)
    actions_buf  = torch.zeros(STEPS, NUM_ENVS, 14,  device=device)
    logprobs_buf = torch.zeros(STEPS, NUM_ENVS,      device=device)
    values_buf   = torch.zeros(STEPS, NUM_ENVS,      device=device)
    rewards_buf  = torch.zeros(STEPS, NUM_ENVS,      device=device)

    cuda_graph       = None
    ppo_graph        = None
    pg_states        = None
    pg_actions       = None
    pg_logp_old      = None
    pg_returns       = None
    pg_adv           = None
    prev_map_hash    = None

    for epoch in range(epochs):
        # ── Curriculum tier ──────────────────────────────────────────────────
        if moving_acc >= 95:                              current_tier_code = 4
        elif moving_acc >= 85 and current_tier_code < 3: current_tier_code = 3
        elif moving_acc >= 70 and current_tier_code < 2: current_tier_code = 2
        if   current_tier_code == 4 and moving_acc < 80: current_tier_code = 3
        elif current_tier_code == 3 and moving_acc < 70: current_tier_code = 2
        elif current_tier_code == 2 and moving_acc < 55: current_tier_code = 1

        easy_maps   = [c['hash'] for c in curriculum if c['nps'] < 2]
        medium_maps = [c['hash'] for c in curriculum if c['nps'] < 4]
        hard_maps   = [c['hash'] for c in curriculum if c['nps'] < 7]
        all_maps    = curriculum_hashes

        if   current_tier_code == 4: tier, primary, fallback = "EXPERT", all_maps,    hard_maps
        elif current_tier_code == 3: tier, primary, fallback = "HARD",   hard_maps,   medium_maps
        elif current_tier_code == 2: tier, primary, fallback = "MEDIUM", medium_maps, easy_maps
        else:                        tier, primary, fallback = "EASY",   easy_maps,   easy_maps

        if not primary:  primary  = curriculum_hashes
        if not fallback: fallback = primary

        lr = optimizer.param_groups[0]['lr']
        print(f"\n\033[95m--- Epoch {epoch+1}/{epochs} | {tier} | LR={lr:.2e} | {NUM_ENVS} envs ---\033[0m")
        t0 = time.time()

        # ── Load maps & Random Start Offsets ──────────────────────────────────
        all_notes, all_bpms = [], []
        n_primary = int(NUM_ENVS * 0.7)
        for i in range(NUM_ENVS):
            pool = primary if i < n_primary else fallback
            h = np.random.choice(pool)
            notes, bpm = map_cache.get(h, ([], 120.0))
            all_notes.append(notes)
            all_bpms.append(bpm)

        sim.load_maps(all_notes, all_bpms)
        
        # Stochastic Start Offsets: ensuring full-map coverage
        max_start = (sim.map_durations - 22.0).clamp(min=0.0)
        start_times = torch.rand(NUM_ENVS, device=device) * max_start

        # ── Build / rebuild CUDA graph ────────────────────────────────────────
        map_shape_key = (sim.max_notes, NUM_ENVS)
        if cuda_graph is None or map_shape_key != prev_map_hash:
            if cuda_graph is not None:
                print("  [CUDA graph] Map shape changed — rebuilding graph.")
            model.eval()
            (cuda_graph, g_mean, g_std, g_value, g_action, g_logp,
             g_state, g_reward, g_noise) = build_cuda_graph(sim, model, device)
            prev_map_hash = map_shape_key
            sim.reset(start_times)
        else:
            sim.reset(start_times)

        # ── Rollout ───────────────────────────────────────────────────────────
        all_noise = torch.randn(STEPS, NUM_ENVS, 14, device=device)
        print(f"  \033[93mReplaying graph {STEPS} times ({STEPS*NUM_ENVS:,} transitions)...\033[0m")

        for step in range(STEPS):
            g_noise.copy_(all_noise[step])
            cuda_graph.replay()
            states_buf[step].copy_(g_state, non_blocking=True)
            actions_buf[step].copy_(g_action, non_blocking=True)
            logprobs_buf[step].copy_(g_logp, non_blocking=True)
            values_buf[step].copy_(g_value.squeeze(-1), non_blocking=True)
            rewards_buf[step].copy_(g_reward, non_blocking=True)

        torch.cuda.synchronize()
        sim_time = time.time() - t0

        # ── Stats ─────────────────────────────────────────────────────────────
        avg_hits   = sim.total_hits.mean().item()
        avg_misses = sim.total_misses.mean().item()
        acc        = (avg_hits / max(1, avg_hits + avg_misses)) * 100
        avg_cut    = sim.total_scores.sum().item() / max(1, sim.total_hits.sum().item())
        moving_acc = 0.85 * moving_acc + 0.15 * acc

        vram_used = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        print(f"\033[96m{'='*55}")
        print(f"  EPOCH {epoch+1} | {NUM_ENVS} envs | {STEPS*NUM_ENVS:,} transitions")
        print(f"{'='*55}\033[0m")
        print(f"  \033[92mAvg Hits:\033[0m       {avg_hits:.1f}")
        print(f"  \033[91mAvg Misses:\033[0m     {avg_misses:.1f}")
        print(f"  \033[93mAccuracy:\033[0m       {acc:.2f}%  (MA: {moving_acc:.2f}%)")
        print(f"  \033[95mCut Score:\033[0m      {avg_cut:.1f} / 115")
        print(f"  \033[90mSim Time:\033[0m       {sim_time:.2f}s  ({STEPS*NUM_ENVS/sim_time:.0f} steps/sec)")
        print(f"  \033[90mPeak VRAM:\033[0m      {vram_used:.2f} GB")
        print(f"\033[96m{'='*55}\033[0m")

        # ── GAE ───────────────────────────────────────────────────────────────
        t_update   = time.time()
        advantages = torch.zeros_like(rewards_buf)
        gae        = torch.zeros(NUM_ENVS, device=device)
        for t in reversed(range(STEPS)):
            next_val = values_buf[t + 1].detach() if t < STEPS - 1 else torch.zeros(NUM_ENVS, device=device)
            delta    = rewards_buf[t] + GAMMA * next_val - values_buf[t].detach()
            gae      = delta + GAMMA * GAE_LAMBDA * gae
            advantages[t] = gae
        returns = advantages + values_buf.detach()

        states_t   = states_buf.reshape(-1, 148)
        actions_t  = actions_buf.reshape(-1, 14)
        logprobs_t = logprobs_buf.reshape(-1).detach()
        returns_t  = returns.reshape(-1)
        adv_t      = advantages.reshape(-1)
        adv_t      = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        adv_t      = adv_t.clamp(-3.0, 3.0)

        # ── PPO update ───────────────────────────────────────────────────────
        model.train()
        PPO_EPOCHS = 2 if epoch < 100 else 4
        BATCH      = 32768
        ds         = states_t.size(0)
 
        if ppo_graph is None:
            (ppo_graph, pg_states, pg_actions, pg_logp_old, pg_returns, pg_adv) = build_ppo_graph(model, optimizer, BATCH, device)
 
        for ppo_epoch in range(PPO_EPOCHS):
            idx = torch.randperm(ds, device=device)
            for i in range(0, ds - BATCH + 1, BATCH):
                b = idx[i:i+BATCH]
                pg_states.copy_(states_t[b])
                pg_actions.copy_(actions_t[b])
                pg_logp_old.copy_(logprobs_t[b])
                pg_returns.copy_(returns_t[b])
                pg_adv.copy_(adv_t[b])
                ppo_graph.replay()
            torch.cuda.synchronize()
            print(f"  \033[90mPPO epoch {ppo_epoch+1}/{PPO_EPOCHS}\033[0m")

        scheduler.step()
        update_time = time.time() - t_update
        print(f"  \033[90mUpdate: {update_time:.2f}s | Total: {time.time()-t0:.2f}s\033[0m")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), model_path)
            with open(STATE_PATH, 'w') as f:
                json.dump({'moving_acc': moving_acc, 'current_tier_code': current_tier_code}, f)
            print(f"\033[92m  Checkpoint saved!\033[0m")

        if (epoch + 1) % 20 == 0 and moving_acc > 25.0 and len(curriculum_hashes) < 720:
            print(f"\033[94m  Expanding curriculum with new ranked maps...\033[0m")
            new_maps = fetch_random_ranked_maps(count=10)
            if new_maps > 0:
                analyze_all_maps()
                with open('curriculum.json') as f:
                    curriculum = json.load(f)
                curriculum_hashes = [c['hash'] for c in curriculum]
                for h in curriculum_hashes:
                    if h not in map_cache:
                        notes, bpm = get_map_notes(h)
                        if notes:
                            map_cache[h] = (notes, bpm)
                curriculum_hashes = list(map_cache.keys())
                cuda_graph    = None
                prev_map_hash = None
                print(f"\033[94m  Curriculum now has {len(curriculum)} maps (+{new_maps} new)\033[0m")


if __name__ == "__main__":
    try:
        train_ppo_gpu()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
