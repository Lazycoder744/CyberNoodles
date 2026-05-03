import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cybernoodles.core.map_storage import slim_map_archive, slim_map_cache
from cybernoodles.paths import PROJECT_ROOT

DATA_DIR = "data"
REPLAYS_DIR = os.path.join(DATA_DIR, "replays")
MAPS_DIR = os.path.join(DATA_DIR, "maps")
PLAYER_CONFIG_PATH = os.path.join(DATA_DIR, "player_ids.json")
PLAYER_SCORE_CACHE_DIR = os.path.join(DATA_DIR, "player_score_caches")
SELECTED_CACHE = os.path.join(DATA_DIR, "selected_scores.json")

DEFAULT_PLAYER_IDS = ["76561199225754020"]
BL_PLAYER_URL = "https://api.beatleader.xyz/player/{player_id}?leaderboardContext=general"
BL_PLAYER_SEARCH_URL = "https://api.beatleader.xyz/players"
BL_SCORES_URL = "https://api.beatleader.xyz/player/{player_id}/scores"
BS_MAP_URL = "https://api.beatsaver.com/maps/hash/{map_hash}"

DEFAULT_MIN_ACCURACY = 0.85
DEFAULT_TOP_N = 5000
DEFAULT_PAGE_SIZE = 100
DEFAULT_DOWNLOAD_WORKERS = 4
DEFAULT_MAP_LOOKUP_WORKERS = 6
DEFAULT_REQUEST_DELAY = 0.05
DEFAULT_MAX_STARS = 12.0
DEFAULT_FETCH_CONFIG_PATH = PROJECT_ROOT / "fetch_data_config.json"
DEFAULT_FETCH_CONFIG = {
    "selection": {
        "min_accuracy": 0.90,
        "top_n": 5000,
        "per_player_limit": None,
        "selected_per_player_limit": None,
        "require_ranked": True,
        "require_standard_mode": True,
        "require_no_mods": True,
        "require_full_combo": True,
        "max_stars": 12.0,
    },
    "download": {
        "page_size": 100,
        "max_pages_per_player": None,
        "request_delay": 0.05,
        "download_workers": 4,
        "map_lookup_workers": 6,
    },
}

_THREAD_LOCAL = threading.local()

os.makedirs(REPLAYS_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(PLAYER_SCORE_CACHE_DIR, exist_ok=True)


def _normalize_text(value):
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _score_is_ranked(score):
    diff = score.get("leaderboard", {}).get("difficulty", {})
    stars = diff.get("stars")
    ranked_time = diff.get("rankedTime") or 0
    pp = float(score.get("pp") or 0.0)
    with_pp = bool(score.get("withPp"))
    return bool(stars or ranked_time or with_pp or pp > 0.0)


def _score_is_standard(score):
    diff = score.get("leaderboard", {}).get("difficulty", {})
    mode_name = _normalize_text(diff.get("modeName"))
    return mode_name in ("standard", "")


def _score_has_supported_mods(score):
    modifiers = str(score.get("modifiers") or "").strip()
    return not modifiers


def _score_is_full_combo(score):
    full_combo = score.get("fullCombo")
    if full_combo is not None:
        return bool(full_combo)
    return int(score.get("missedNotes") or 0) <= 0 and int(score.get("badCuts") or 0) <= 0


def _score_stars(score):
    diff = score.get("leaderboard", {}).get("difficulty", {})
    stars = diff.get("stars")
    if stars is None:
        return None
    try:
        return float(stars)
    except (TypeError, ValueError):
        return None


def _format_bytes(num_bytes):
    return f"{num_bytes / (1024 ** 2):.1f} MB"


def build_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "CyberNoodles-DatasetFetcher/2.0"})
    return session


def get_thread_session():
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = build_session()
        _THREAD_LOCAL.session = session
    return session


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, payload):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, path)


def _clone_default_config():
    return json.loads(json.dumps(DEFAULT_FETCH_CONFIG))


def _merge_config(base, override):
    if not isinstance(override, dict):
        return base
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value
    return base


def load_fetch_config(path):
    config = _clone_default_config()
    raw = load_json(path, {})
    if isinstance(raw, dict):
        _merge_config(config, raw)
    return config


def load_selected_cache(path=SELECTED_CACHE):
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        return None
    selected = payload.get("selected")
    if not isinstance(selected, list):
        return None
    return payload


def resolve_option(cli_value, config_value, default_value):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default_value


def coerce_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off"):
            return False
    return bool(value)


def coerce_int(value, default, allow_none=False):
    if value is None:
        return None if allow_none else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return None if allow_none else default


def coerce_float(value, default, allow_none=False):
    if value is None:
        return None if allow_none else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return None if allow_none else default


def player_cache_path(player_id):
    return os.path.join(PLAYER_SCORE_CACHE_DIR, f"{player_id}.json")


def load_player_config():
    payload = load_json(PLAYER_CONFIG_PATH, {})
    players = payload.get("player_ids")
    if isinstance(players, list):
        return [str(p).strip() for p in players if str(p).strip()]
    return []


def save_player_config(player_ids):
    save_json(PLAYER_CONFIG_PATH, {"player_ids": list(player_ids)})


def parse_player_reference(ref):
    text = str(ref or "").strip()
    if not text:
        return None
    if text.isdigit():
        return text

    patterns = [
        r"steamcommunity\.com/profiles/(\d+)",
        r"beatleader\.[^/]+/u/(\d+)",
        r"api\.beatleader\.[^/]+/player/(\d+)",
        r"/player/(\d+)",
        r"/u/(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def search_players_by_name(session, player_name, count=10):
    response = session.get(
        BL_PLAYER_SEARCH_URL,
        params={"search": player_name, "count": max(1, min(50, int(count)))},
        timeout=30,
    )
    if response.status_code != 200:
        return []
    payload = response.json()
    results = payload.get("data", [])
    return results if isinstance(results, list) else []


def resolve_player_name(session, player_name):
    name = str(player_name or "").strip()
    if not name:
        return None

    try:
        results = search_players_by_name(session, name, count=10)
    except Exception:
        return None

    if not results:
        return None

    normalized_query = _normalize_text(name)
    exact_matches = []
    for item in results:
        for field in ("name", "alias"):
            if _normalize_text(item.get(field)) == normalized_query:
                exact_matches.append(item)
                break

    matches = exact_matches or results
    chosen = matches[0]
    player_id = str(chosen.get("id") or "").strip()
    if not player_id:
        return None

    return {
        "id": player_id,
        "name": chosen.get("name") or name,
        "country": chosen.get("country"),
        "rank": chosen.get("rank"),
        "pp": chosen.get("pp"),
        "match_type": "exact" if exact_matches else "search",
    }


def resolve_player_reference(session, ref):
    raw = str(ref or "").strip()
    if not raw:
        return None

    direct_id = parse_player_reference(raw)
    if direct_id is not None:
        return {
            "id": direct_id,
            "name": None,
            "match_type": "direct",
            "input": raw,
        }

    resolved = resolve_player_name(session, raw)
    if resolved is None:
        return None
    resolved["input"] = raw
    return resolved


def read_players_file(path):
    refs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            refs.extend(part.strip() for part in raw.split(",") if part.strip())
    return refs


def prompt_player_references():
    raw = input(
        "Enter BeatLeader/Steam player IDs, usernames, or profile URLs "
        "(comma-separated, only with permission): "
    ).strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_player_ids(args):
    refs = []
    refs.extend(args.player_id or [])
    refs.extend(args.player_name or [])
    refs.extend(args.player_url or [])
    if args.players_file:
        refs.extend(read_players_file(args.players_file))
    if args.interactive:
        refs.extend(prompt_player_references())
    if not refs:
        refs.extend(load_player_config() or DEFAULT_PLAYER_IDS)

    session = build_session()
    player_ids = []
    seen = set()
    for ref in refs:
        resolved = resolve_player_reference(session, ref)
        if resolved is None:
            print(f"Skipping unrecognized player reference or username: {ref}")
            continue
        player_id = resolved["id"]
        if resolved.get("match_type") in ("exact", "search"):
            rank = resolved.get("rank")
            rank_text = f", rank {rank}" if rank is not None else ""
            print(
                f"Resolved username '{resolved['input']}' -> "
                f"{resolved.get('name') or player_id} ({player_id}{rank_text})"
            )
        if player_id in seen:
            continue
        seen.add(player_id)
        player_ids.append(player_id)

    if not player_ids:
        player_ids = list(DEFAULT_PLAYER_IDS)

    save_player_config(player_ids)
    return player_ids


def fetch_player_profile(session, player_id):
    url = BL_PLAYER_URL.format(player_id=player_id)
    response = session.get(url, timeout=30)
    if response.status_code != 200:
        return {"id": player_id, "name": player_id, "status": response.status_code}
    payload = response.json()
    return {
        "id": player_id,
        "name": payload.get("name") or payload.get("playerName") or player_id,
        "country": payload.get("country"),
        "externalProfileUrl": payload.get("externalProfileUrl"),
        "linkedIds": payload.get("linkedIds", {}),
        "status": response.status_code,
    }


def load_cached_scores(player_id):
    cache_path = player_cache_path(player_id)
    payload = load_json(cache_path, [])
    meta = {
        "complete_history": False,
        "schema_version": 1,
        "total_scores_hint": None,
    }

    if isinstance(payload, dict):
        scores = payload.get("scores", [])
        if not isinstance(scores, list):
            scores = []
        meta["complete_history"] = bool(payload.get("complete_history", False))
        meta["schema_version"] = int(payload.get("schema_version", 2) or 2)
        total_scores_hint = payload.get("total_scores_hint")
        try:
            meta["total_scores_hint"] = int(total_scores_hint) if total_scores_hint is not None else None
        except (TypeError, ValueError):
            meta["total_scores_hint"] = None
    elif isinstance(payload, list):
        # Legacy list-only caches may be incomplete. Treat them as partial so
        # later dry-runs/full fetches keep walking older pages instead of
        # incorrectly stopping on page 1 after a capped probe.
        scores = payload
    else:
        scores = []

    return scores, {str(score.get("id")) for score in scores if score.get("id") is not None}, meta


def save_cached_scores(player_id, scores, complete_history=False, total_scores_hint=None):
    payload = {
        "schema_version": 2,
        "complete_history": bool(complete_history),
        "total_scores_hint": int(total_scores_hint) if total_scores_hint is not None else None,
        "scores": list(scores),
    }
    save_json(player_cache_path(player_id), payload)


def dedupe_scores(scores):
    merged = {}
    for score in scores:
        score_id = str(score.get("id"))
        if score_id == "None":
            continue
        merged[score_id] = score
    values = list(merged.values())
    values.sort(key=lambda score: score.get("timepost", 0) or 0, reverse=True)
    return values


def fetch_scores_for_player(session, player_id, page_size, request_delay, max_pages=None):
    cached_scores, known_ids, cache_meta = load_cached_scores(player_id)
    cache_hit = bool(cached_scores)
    cache_complete = bool(cache_meta.get("complete_history", False))
    if cache_hit and cache_complete:
        print(f"Loaded complete cache for player {player_id} ({len(cached_scores)} scores). Checking for new scores...")
    elif cache_hit:
        print(f"Loaded partial cache for player {player_id} ({len(cached_scores)} scores). Extending history...")
    else:
        print(f"No cache for player {player_id}. Fetching full score history...")

    new_scores = []
    page = 1
    done = False
    hit_error = False
    reached_end = False
    total_scores_hint = cache_meta.get("total_scores_hint")
    total_pages = None
    last_page_fetched = 0
    while not done:
        if max_pages is not None and page > max_pages:
            break
        url = (
            f"{BL_SCORES_URL.format(player_id=player_id)}"
            f"?sortBy=date&order=desc&page={page}&count={page_size}"
        )
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            print(f"  Failed to fetch page {page} for player {player_id}: {response.status_code}")
            hit_error = True
            break

        payload = response.json()
        page_scores = payload.get("data", [])
        if not page_scores:
            reached_end = True
            break
        last_page_fetched = page

        total = payload.get("metadata", {}).get("total")
        try:
            total_scores_hint = int(total) if total is not None else total_scores_hint
        except (TypeError, ValueError):
            total_scores_hint = total_scores_hint
        if total_scores_hint is not None:
            total_pages = max(1, (total_scores_hint + page_size - 1) // page_size)

        for score in page_scores:
            score_id = str(score.get("id"))
            if score_id in known_ids:
                if cache_complete:
                    done = True
                    break
                continue
            new_scores.append(score)

        total_text = f"/{total_scores_hint}" if total_scores_hint is not None else ""
        print(
            f"  Player {player_id}: page {page} "
            f"({len(page_scores)} scores, {len(new_scores)} new{total_text})"
        )
        if total_pages is not None and page >= total_pages:
            reached_end = True
            break
        page += 1
        time.sleep(request_delay)

    merged_scores = dedupe_scores(new_scores + cached_scores)
    if new_scores or not cache_hit or reached_end:
        complete_history = False
        if cache_complete and not hit_error:
            complete_history = True
        elif not hit_error and reached_end:
            complete_history = True
        save_cached_scores(
            player_id,
            merged_scores,
            complete_history=complete_history,
            total_scores_hint=total_scores_hint if total_scores_hint is not None else len(merged_scores),
        )

    return merged_scores, {
        "cached": len(cached_scores),
        "new": len(new_scores),
        "total": len(merged_scores),
    }


def score_to_candidate(score, player_id, player_name):
    leaderboard = score.get("leaderboard", {})
    song = leaderboard.get("song", {})
    difficulty = leaderboard.get("difficulty", {})
    score_id = str(score.get("id"))
    return {
        "id": score_id,
        "score_id": score_id,
        "player_id": str(player_id),
        "player_name": player_name,
        "accuracy": float(score.get("accuracy") or 0.0),
        "pp": float(score.get("pp") or 0.0),
        "rank": score.get("rank"),
        "timepost": score.get("timepost"),
        "leaderboard_id": str(leaderboard.get("id") or ""),
        "replay": score.get("replay"),
        "modifiers": str(score.get("modifiers") or ""),
        "full_combo": _score_is_full_combo(score),
        "map_hash": str(song.get("hash") or "").lower(),
        "song_name": song.get("name") or "?",
        "song_author": song.get("author") or "",
        "mapper": song.get("mapper") or "",
        "difficulty": difficulty.get("difficultyName") or difficulty.get("value") or "",
        "mode": difficulty.get("modeName") or "",
        "stars": _score_stars(score),
    }


def _candidate_accuracy(candidate):
    return float(candidate.get("accuracy") or 0.0)


def _top_n_limit(top_n):
    return None if top_n is None else max(0, int(top_n))


def _select_with_player_cap(candidates, top_n, selected_per_player_limit=None):
    ordered = sorted(candidates, key=_candidate_accuracy, reverse=True)
    player_count = len({str(item.get("player_id") or "") for item in ordered})
    limit = _top_n_limit(top_n)
    if limit == 0:
        cap = max(1, int(selected_per_player_limit)) if selected_per_player_limit is not None else None
        return [], {
            "enabled": cap is not None,
            "eligible_players": player_count,
            "selected_per_player_limit": cap,
            "capped_candidates": 0,
            "filled_over_cap": 0,
        }
    if selected_per_player_limit is None:
        selected = ordered[:limit] if limit is not None else ordered
        return selected, {
            "enabled": False,
            "eligible_players": player_count,
            "selected_per_player_limit": None,
            "capped_candidates": 0,
            "filled_over_cap": 0,
        }

    cap = max(1, int(selected_per_player_limit))

    selected = []
    selected_ids = set()
    counts = {}
    capped_candidates = 0
    for candidate in ordered:
        player_id = str(candidate.get("player_id") or "")
        if counts.get(player_id, 0) >= cap:
            capped_candidates += 1
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        counts[player_id] = counts.get(player_id, 0) + 1
        if limit is not None and len(selected) >= limit:
            break

    filled_over_cap = 0
    if limit is not None and len(selected) < limit:
        for candidate in ordered:
            if id(candidate) in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(id(candidate))
            filled_over_cap += 1
            if len(selected) >= limit:
                break

    return selected, {
        "enabled": True,
        "eligible_players": player_count,
        "selected_per_player_limit": cap,
        "capped_candidates": capped_candidates,
        "filled_over_cap": filled_over_cap,
    }


def select_top_scores(
    scores_by_player,
    profiles,
    min_accuracy,
    top_n,
    per_player_limit,
    require_ranked,
    require_standard_mode,
    require_no_mods,
    require_full_combo,
    max_stars,
    selected_per_player_limit=None,
):
    selected = []
    stats = {
        "total_scores": 0,
        "accuracy_filtered": 0,
        "ranked_only": 0,
        "standard_only": 0,
        "mod_clean": 0,
        "full_combo_only": 0,
        "stars_capped": 0,
        "selected_total": 0,
        "player_cap": {},
        "per_player": {},
    }

    for player_id, scores in scores_by_player.items():
        profile = profiles.get(player_id, {"name": player_id})
        player_candidates = []
        local_stats = {
            "total_scores": len(scores),
            "accuracy_filtered": 0,
            "ranked_only": 0,
            "standard_only": 0,
            "mod_clean": 0,
            "full_combo_only": 0,
            "stars_capped": 0,
            "candidate_count": 0,
            "selected": 0,
        }
        for score in scores:
            stats["total_scores"] += 1
            if float(score.get("accuracy") or 0.0) < min_accuracy:
                continue
            local_stats["accuracy_filtered"] += 1
            stats["accuracy_filtered"] += 1

            if require_ranked and not _score_is_ranked(score):
                continue
            local_stats["ranked_only"] += 1
            stats["ranked_only"] += 1

            if require_standard_mode and not _score_is_standard(score):
                continue
            local_stats["standard_only"] += 1
            stats["standard_only"] += 1

            if require_no_mods and not _score_has_supported_mods(score):
                continue
            local_stats["mod_clean"] += 1
            stats["mod_clean"] += 1

            if require_full_combo and not _score_is_full_combo(score):
                continue
            local_stats["full_combo_only"] += 1
            stats["full_combo_only"] += 1

            score_stars = _score_stars(score)
            if max_stars is not None and (score_stars is None or score_stars > max_stars):
                continue
            local_stats["stars_capped"] += 1
            stats["stars_capped"] += 1

            candidate = score_to_candidate(score, player_id, profile.get("name", player_id))
            if not candidate["map_hash"]:
                continue
            player_candidates.append(candidate)

        player_candidates.sort(key=lambda item: item["accuracy"], reverse=True)
        if per_player_limit is not None:
            player_candidates = player_candidates[:per_player_limit]
        local_stats["candidate_count"] = len(player_candidates)
        stats["per_player"][player_id] = {
            "player_name": profile.get("name", player_id),
            **local_stats,
        }
        selected.extend(player_candidates)

    selected, cap_stats = _select_with_player_cap(
        selected,
        top_n,
        selected_per_player_limit=selected_per_player_limit,
    )

    stats["selected_total"] = len(selected)
    stats["player_cap"] = cap_stats
    final_counts = {}
    for item in selected:
        player_id = str(item.get("player_id") or "")
        final_counts[player_id] = final_counts.get(player_id, 0) + 1
    for player_id, payload in stats["per_player"].items():
        payload["selected"] = final_counts.get(player_id, 0)

    save_json(
        SELECTED_CACHE,
        {
            "filters": {
                "min_accuracy": min_accuracy,
                "top_n": top_n,
                "per_player_limit": per_player_limit,
                "selected_per_player_limit": selected_per_player_limit,
                "require_ranked": require_ranked,
                "require_standard_mode": require_standard_mode,
                "require_no_mods": require_no_mods,
                "require_full_combo": require_full_combo,
                "max_stars": max_stars,
            },
            "players": profiles,
            "stats": stats,
            "selected": selected,
        },
    )
    return selected, stats


def print_selection_stats(scores_by_player, selected, stats):
    print("\nScore breakdown:")
    print(f"  Players loaded:      {len(scores_by_player)}")
    print(f"  Total in cache:      {stats['total_scores']}")
    print(f"  Accuracy filtered:   {stats['accuracy_filtered']}")
    print(f"  Ranked only:         {stats['ranked_only']}")
    print(f"  Standard only:       {stats['standard_only']}")
    print(f"  No gameplay mods:    {stats['mod_clean']}")
    print(f"  Full combo only:     {stats['full_combo_only']}")
    print(f"  Stars cap kept:      {stats['stars_capped']}")
    print(f"  Selected final set:  {stats['selected_total']}")
    player_cap = stats.get("player_cap", {})
    if player_cap.get("enabled") and player_cap.get("selected_per_player_limit") is not None:
        print(
            "  Player cap:          "
            f"{player_cap['selected_per_player_limit']} selected score(s) per player "
            f"across {player_cap.get('eligible_players', 0)} eligible player(s)"
        )
    if selected:
        accuracies = [item["accuracy"] for item in selected]
        print(f"  Accuracy range:      {min(accuracies):.2%} - {max(accuracies):.2%}")
    for player_id, payload in stats["per_player"].items():
        print(
            f"    {payload['player_name']} ({player_id}): "
            f"{payload['selected']} selected from {payload['total_scores']} cached scores"
        )


def download_file(url, path, timeout=60, postprocess=None):
    path_obj = Path(path)
    if path_obj.exists():
        if postprocess is not None:
            postprocess(str(path_obj))
        return "skipped"

    temp_path = path_obj.with_suffix(path_obj.suffix + ".download")
    try:
        session = get_thread_session()
        with session.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                return f"failed:{response.status_code}"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(temp_path, path_obj)
        if postprocess is not None:
            postprocess(str(path_obj))
        return "downloaded"
    except Exception as exc:
        return f"error:{exc}"
    finally:
        if temp_path.exists():
            temp_path.unlink()


def slim_map_postprocess(path):
    changed, before_bytes, after_bytes = slim_map_archive(path)
    return changed, before_bytes, after_bytes


def get_beatsaver_map_url(session, map_hash):
    try:
        if session is None:
            session = get_thread_session()
        response = session.get(BS_MAP_URL.format(map_hash=map_hash), timeout=30)
        if response.status_code != 200:
            return None
        payload = response.json()
        for version in payload.get("versions", []):
            if str(version.get("hash") or "").lower() == map_hash.lower():
                return version.get("downloadURL")
        versions = payload.get("versions") or []
        if versions:
            return versions[0].get("downloadURL")
    except Exception as exc:
        print(f"  Error resolving map {map_hash}: {exc}")
    return None


def download_selected_assets(selected, workers, map_lookup_workers=None):
    replay_tasks = []
    map_tasks = {}
    missing_replay_url = 0
    workers = max(1, int(workers))
    map_lookup_workers = max(1, int(map_lookup_workers or workers))

    for item in selected:
        replay_url = item.get("replay")
        replay_path = os.path.join(REPLAYS_DIR, f"{item['id']}.bsor")
        if replay_url:
            replay_tasks.append((item, replay_url, replay_path))
        else:
            missing_replay_url += 1

        map_hash = item.get("map_hash")
        if not map_hash:
            continue
        map_path = os.path.join(MAPS_DIR, f"{map_hash}.zip")
        if map_hash not in map_tasks:
            map_tasks[map_hash] = {
                "hash": map_hash,
                "path": map_path,
                "song_name": item.get("song_name", "?"),
                "url": None,
            }

    stats = {
        "downloaded_replays": 0,
        "skipped_replays": 0,
        "failed_replays": 0,
        "downloaded_maps": 0,
        "skipped_maps": 0,
        "failed_maps": 0,
        "missing_replay_url": missing_replay_url,
    }

    maps_to_resolve = []
    for map_hash, task in map_tasks.items():
        if os.path.exists(task["path"]):
            changed, before_bytes, after_bytes = slim_map_archive(task["path"])
            if changed:
                print(
                    f"  Slimmed cached map {map_hash} by "
                    f"{_format_bytes(before_bytes - after_bytes)}"
                )
            continue
        maps_to_resolve.append(task)

    if maps_to_resolve:
        print(
            f"\nResolving BeatSaver URLs for {len(maps_to_resolve)} maps "
            f"with {map_lookup_workers} worker(s)..."
        )

        def resolve_map_task(task):
            return task, get_beatsaver_map_url(None, task["hash"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=map_lookup_workers) as executor:
            resolve_futures = [executor.submit(resolve_map_task, task) for task in maps_to_resolve]
            resolved_count = 0
            resolved_ok = 0
            progress_interval = max(10, min(100, len(maps_to_resolve) // 20 if len(maps_to_resolve) > 0 else 10))
            for future in concurrent.futures.as_completed(resolve_futures):
                task, url = future.result()
                task["url"] = url
                resolved_count += 1
                if url:
                    resolved_ok += 1
                if resolved_count == len(maps_to_resolve) or (resolved_count % progress_interval) == 0:
                    print(
                        f"  Resolved {resolved_count}/{len(maps_to_resolve)} maps "
                        f"({resolved_ok} with download URLs)"
                    )

    map_download_tasks = []
    for task in map_tasks.values():
        if os.path.exists(task["path"]):
            stats["skipped_maps"] += 1
            continue
        if not task.get("url"):
            stats["failed_maps"] += 1
            print(f"  Map failed for {task['song_name']} ({task['hash']}): missing-url")
            continue
        map_download_tasks.append(task)

    print(
        f"\nDownloading {len(replay_tasks)} replay candidates and "
        f"{len(map_download_tasks)} map archives "
        f"({stats['skipped_maps']} maps already cached)...\n"
    )

    def run_replay_task(task):
        item, replay_url, replay_path = task
        result = download_file(replay_url, replay_path)
        return item, replay_path, result

    def run_map_task(task):
        if not task["url"]:
            return task, "missing-url", None
        result = download_file(task["url"], task["path"], postprocess=slim_map_postprocess)
        return task, result, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for task in replay_tasks:
            futures[executor.submit(run_replay_task, task)] = "replay"
        for task in map_download_tasks:
            futures[executor.submit(run_map_task, task)] = "map"

        total_replays = len(replay_tasks)
        total_maps = len(map_download_tasks)
        replay_done = 0
        map_done = 0
        replay_progress_interval = max(25, min(250, total_replays // 20 if total_replays > 0 else 25))
        map_progress_interval = max(10, min(100, total_maps // 20 if total_maps > 0 else 10))

        for future in concurrent.futures.as_completed(futures):
            task_type = futures[future]
            if task_type == "replay":
                item, replay_path, result = future.result()
                replay_done += 1
                if result == "downloaded":
                    stats["downloaded_replays"] += 1
                elif result == "skipped":
                    stats["skipped_replays"] += 1
                else:
                    stats["failed_replays"] += 1
                    print(f"  Replay failed for {item['song_name']} ({item['id']}): {result}")
                if replay_done == total_replays or (replay_done % replay_progress_interval) == 0:
                    print(
                        f"  Replays {replay_done}/{total_replays} | "
                        f"downloaded {stats['downloaded_replays']} | "
                        f"skipped {stats['skipped_replays']} | "
                        f"failed {stats['failed_replays']}"
                    )
            else:
                task, result, _ = future.result()
                map_done += 1
                if result == "downloaded":
                    stats["downloaded_maps"] += 1
                elif result == "skipped":
                    stats["skipped_maps"] += 1
                else:
                    stats["failed_maps"] += 1
                    print(f"  Map failed for {task['song_name']} ({task['hash']}): {result}")
                if map_done == total_maps or (map_done % map_progress_interval) == 0:
                    print(
                        f"  Maps {map_done}/{total_maps} | "
                        f"downloaded {stats['downloaded_maps']} | "
                        f"cached {stats['skipped_maps']} | "
                        f"failed {stats['failed_maps']}"
                    )

    cache_stats = slim_map_cache(MAPS_DIR)
    return stats, cache_stats


def fetch_random_ranked_maps(count=10):
    """Download random ranked maps from BeatSaver for RL curriculum expansion."""
    import random

    os.makedirs(MAPS_DIR, exist_ok=True)
    existing = {
        name.replace(".zip", "").lower()
        for name in os.listdir(MAPS_DIR)
        if name.lower().endswith(".zip")
    }
    downloaded = 0
    page = random.randint(0, 50)
    session = build_session()

    print(f"  Fetching random ranked maps from BeatSaver (page {page})...")
    try:
        url = f"https://api.beatsaver.com/search/text/{page}?ranked=true&sortOrder=Relevance&pageSize=20"
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            print(f"  BeatSaver API returned {response.status_code}")
            return 0

        maps_data = response.json().get("docs", [])
        random.shuffle(maps_data)
        for payload in maps_data:
            if downloaded >= count:
                break
            versions = payload.get("versions") or []
            if not versions:
                continue
            map_hash = str(versions[0].get("hash") or "").upper()
            download_url = versions[0].get("downloadURL")
            if not map_hash or not download_url or map_hash.lower() in existing:
                continue

            map_path = os.path.join(MAPS_DIR, f"{map_hash}.zip")
            result = download_file(download_url, map_path, postprocess=slim_map_postprocess)
            if result == "downloaded":
                downloaded += 1
                print(
                    f"    [{downloaded}/{count}] "
                    f"{payload.get('metadata', {}).get('songName', 'Unknown')} "
                    f"({map_hash[:8]}...)"
                )
            time.sleep(DEFAULT_REQUEST_DELAY)
    except Exception as exc:
        print(f"  Error fetching from BeatSaver: {exc}")

    cache_stats = slim_map_cache(MAPS_DIR)
    print(f"  Downloaded {downloaded} new ranked maps for curriculum expansion.")
    print(
        f"  Map cache slimming saved {_format_bytes(cache_stats['saved_bytes'])} "
        f"across {cache_stats['trimmed']} archives."
    )
    return downloaded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download BeatLeader replay corpora for one or more consented players."
    )
    parser.add_argument("--config", default=str(DEFAULT_FETCH_CONFIG_PATH), help="Path to the fetch-data config JSON.")
    parser.add_argument("--from-selected-cache", action="store_true", help="Skip player/API refresh and download directly from data/selected_scores.json.")
    parser.add_argument("--player-id", action="append", help="BeatLeader/Steam player ID. Repeat for multiple players.")
    parser.add_argument("--player-name", action="append", help="BeatLeader username. Repeat for multiple players.")
    parser.add_argument("--player-url", action="append", help="BeatLeader or Steam profile URL. Repeat for multiple players.")
    parser.add_argument("--players-file", help="Text file with one player ID/URL per line.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for player IDs, usernames, or URLs.")
    parser.add_argument("--min-accuracy", type=float, default=None, help="Minimum replay accuracy to keep (0-1).")
    parser.add_argument("--top-n", type=int, default=None, help="Global number of top replays to keep after filtering.")
    parser.add_argument("--per-player-limit", type=int, default=None, help="Optional per-player cap before global top-N selection.")
    parser.add_argument("--selected-per-player-limit", type=int, default=None, help="Optional per-player cap applied during final selected-score top-N selection.")
    parser.add_argument("--max-stars", type=float, default=None, help="Keep only scores at or below this star value.")
    parser.add_argument("--require-ranked", action=argparse.BooleanOptionalAction, default=None, help="Require ranked scores.")
    parser.add_argument("--require-standard-mode", action=argparse.BooleanOptionalAction, default=None, help="Require standard mode scores.")
    parser.add_argument("--require-no-mods", action=argparse.BooleanOptionalAction, default=None, help="Require no gameplay modifiers.")
    parser.add_argument("--require-full-combo", action=argparse.BooleanOptionalAction, default=None, help="Require full-combo scores.")
    parser.add_argument("--allow-non-standard", dest="require_standard_mode", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--allow-mods", dest="require_no_mods", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--page-size", type=int, default=None, help="BeatLeader page size per request.")
    parser.add_argument("--max-pages-per-player", type=int, default=None, help="Optional safety cap on pages fetched per player.")
    parser.add_argument("--request-delay", type=float, default=None, help="Delay between BeatLeader page fetches.")
    parser.add_argument("--download-workers", type=int, default=None, help="Concurrent download workers.")
    parser.add_argument("--map-lookup-workers", type=int, default=None, help="Concurrent BeatSaver map lookup workers.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch metadata and selection manifests without downloading assets.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Only download replays for players who explicitly allowed you to use their data.")

    config_path = Path(args.config).resolve()
    config = load_fetch_config(str(config_path))
    selection_config = config.get("selection", {})
    download_config = config.get("download", {})

    min_accuracy = max(
        0.0,
        min(
            1.0,
            coerce_float(
                resolve_option(args.min_accuracy, selection_config.get("min_accuracy"), DEFAULT_MIN_ACCURACY),
                DEFAULT_MIN_ACCURACY,
            ),
        ),
    )
    top_n = coerce_int(resolve_option(args.top_n, selection_config.get("top_n"), DEFAULT_TOP_N), DEFAULT_TOP_N, allow_none=True)
    per_player_limit = coerce_int(
        resolve_option(args.per_player_limit, selection_config.get("per_player_limit"), None),
        None,
        allow_none=True,
    )
    selected_per_player_limit = coerce_int(
        resolve_option(
            args.selected_per_player_limit,
            selection_config.get("selected_per_player_limit"),
            None,
        ),
        None,
        allow_none=True,
    )
    max_stars = coerce_float(
        resolve_option(args.max_stars, selection_config.get("max_stars"), DEFAULT_MAX_STARS),
        DEFAULT_MAX_STARS,
        allow_none=True,
    )
    require_ranked = coerce_bool(
        resolve_option(args.require_ranked, selection_config.get("require_ranked"), True),
        True,
    )
    require_standard_mode = coerce_bool(
        resolve_option(args.require_standard_mode, selection_config.get("require_standard_mode"), True),
        True,
    )
    require_no_mods = coerce_bool(
        resolve_option(args.require_no_mods, selection_config.get("require_no_mods"), True),
        True,
    )
    require_full_combo = coerce_bool(
        resolve_option(args.require_full_combo, selection_config.get("require_full_combo"), False),
        False,
    )
    page_size = max(
        1,
        min(
            100,
            coerce_int(resolve_option(args.page_size, download_config.get("page_size"), DEFAULT_PAGE_SIZE), DEFAULT_PAGE_SIZE),
        ),
    )
    max_pages_per_player = coerce_int(
        resolve_option(args.max_pages_per_player, download_config.get("max_pages_per_player"), None),
        None,
        allow_none=True,
    )
    request_delay = max(
        0.0,
        coerce_float(resolve_option(args.request_delay, download_config.get("request_delay"), DEFAULT_REQUEST_DELAY), DEFAULT_REQUEST_DELAY),
    )
    download_workers = max(
        1,
        coerce_int(
            resolve_option(args.download_workers, download_config.get("download_workers"), DEFAULT_DOWNLOAD_WORKERS),
            DEFAULT_DOWNLOAD_WORKERS,
        ),
    )
    map_lookup_workers = max(
        1,
        coerce_int(
            resolve_option(args.map_lookup_workers, download_config.get("map_lookup_workers"), DEFAULT_MAP_LOOKUP_WORKERS),
            DEFAULT_MAP_LOOKUP_WORKERS,
        ),
    )

    print(f"Fetch config: {config_path}")

    if args.from_selected_cache:
        payload = load_selected_cache()
        if payload is None:
            print(f"No usable selected replay manifest found at {SELECTED_CACHE}.")
            print("Run a normal dry-run first to build data/selected_scores.json.")
            return

        selected = payload.get("selected", [])
        stats = payload.get("stats", {})
        filters = payload.get("filters", {})
        print(
            f"Using cached selection manifest: {len(selected)} replay(s) "
            f"from {SELECTED_CACHE}"
        )
        if filters:
            print(f"Cached filters: {json.dumps(filters, sort_keys=True)}")
        if stats:
            print(f"Cached selected count: {stats.get('selected_total', len(selected))}")

        if not selected:
            print("Selected replay manifest is empty. Run a normal dry-run with your players first.")
            return

        if args.dry_run:
            print("\nDry run complete. Cached selection manifest is ready for direct download.")
            return

        asset_stats, cache_stats = download_selected_assets(
            selected,
            workers=download_workers,
            map_lookup_workers=map_lookup_workers,
        )
        print("\nDone!")
        print(
            f"  Replays:  {asset_stats['downloaded_replays']} downloaded, "
            f"{asset_stats['skipped_replays']} already existed, "
            f"{asset_stats['failed_replays']} failed, "
            f"{asset_stats['missing_replay_url']} had no replay URL"
        )
        print(
            f"  Maps:     {asset_stats['downloaded_maps']} downloaded, "
            f"{asset_stats['skipped_maps']} already existed, "
            f"{asset_stats['failed_maps']} failed"
        )
        print(
            f"  Map cache slimming: {cache_stats['trimmed']} archives rewritten, "
            f"saved {_format_bytes(cache_stats['saved_bytes'])}"
        )
        return

    player_ids = resolve_player_ids(args)
    session = build_session()

    profiles = {}
    scores_by_player = {}
    for player_id in player_ids:
        profile = fetch_player_profile(session, player_id)
        profiles[player_id] = profile
        print(f"\nPlayer {profile['name']} ({player_id})")
        if profile.get("externalProfileUrl"):
            print(f"  External profile: {profile['externalProfileUrl']}")
        scores, cache_stats = fetch_scores_for_player(
            session,
            player_id,
            page_size=page_size,
            request_delay=request_delay,
            max_pages=max_pages_per_player,
        )
        print(
            f"  Cached scores: {cache_stats['cached']} | "
            f"new: {cache_stats['new']} | total: {cache_stats['total']}"
        )
        scores_by_player[player_id] = scores

    selected, stats = select_top_scores(
        scores_by_player,
        profiles,
        min_accuracy=min_accuracy,
        top_n=max(1, top_n) if top_n is not None else None,
        per_player_limit=max(1, per_player_limit) if per_player_limit is not None else None,
        require_ranked=require_ranked,
        require_standard_mode=require_standard_mode,
        require_no_mods=require_no_mods,
        require_full_combo=require_full_combo,
        max_stars=max_stars,
        selected_per_player_limit=max(1, selected_per_player_limit) if selected_per_player_limit is not None else None,
    )
    print_selection_stats(scores_by_player, selected, stats)

    if not selected:
        print("No qualifying scores found. Try lowering the replay filters in fetch_data_config.json.")
        return

    if args.dry_run:
        print("\nDry run complete. Selection manifest written to data/selected_scores.json.")
        return

    asset_stats, cache_stats = download_selected_assets(
        selected,
        workers=download_workers,
        map_lookup_workers=map_lookup_workers,
    )
    print("\nDone!")
    print(
        f"  Replays:  {asset_stats['downloaded_replays']} downloaded, "
        f"{asset_stats['skipped_replays']} already existed, "
        f"{asset_stats['failed_replays']} failed, "
        f"{asset_stats['missing_replay_url']} had no replay URL"
    )
    print(
        f"  Maps:     {asset_stats['downloaded_maps']} downloaded, "
        f"{asset_stats['skipped_maps']} already existed, "
        f"{asset_stats['failed_maps']} failed"
    )
    print(
        f"  Map cache slimming: {cache_stats['trimmed']} archives rewritten, "
        f"saved {_format_bytes(cache_stats['saved_bytes'])}"
    )


if __name__ == "__main__":
    main()
