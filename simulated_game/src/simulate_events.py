import random
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import csv
import uuid

# ----------------------------
# Config objects
# ----------------------------

@dataclass
class DifficultyConfig:
    base_pass_prob_after_pass: float = 0.80  # P(pass | last was pass) at level 1
    base_pass_prob_after_fail: float = 0.70  # P(pass | last was fail) at level 1
    difficulty_drop_over_game: float = 0.20  # how much to reduce pass chance by level 100

    def adjusted_probs(self, level: int, max_level: int) -> Dict[str, float]:
        """
        Linearly reduce pass probability as level increases.
        At level 1: no difficulty penalty.
        At level max_level: full 'difficulty_drop_over_game' penalty.
        """
        # Fraction through the game from 0 (level1) to 1 (max_level)
        frac = (level - 1) / (max_level - 1)
        penalty = frac * self.difficulty_drop_over_game

        return {
            "after_pass": max(0.0, min(1.0, self.base_pass_prob_after_pass - penalty)),
            "after_fail": max(0.0, min(1.0, self.base_pass_prob_after_fail - penalty)),
        }


@dataclass
class DeviceConfig:
    device_types: List[str] = None   # e.g. ["tablet", "phone", "desktop"]
    stick_with_prev_prob: float = 0.95  # P(same device as last time)

    def __post_init__(self):
        if self.device_types is None:
            self.device_types = ["tablet", "phone", "desktop"]


@dataclass
class CompletionConfig:
    complete_prob: float = 0.80  # P(complete vs cancel)

@dataclass
class ProgressionConfig:
    immediate_next_level_prob: float = 0.70
    # Probabilities when NOT immediate:
    next_day_prob: float = 0.50
    next_week_prob: float = 0.30
    next_month_prob: float = 0.20

@dataclass
class SessionDurationConfig:
    # Minutes for completed attempts
    completed_min_minutes: float = 2.0
    completed_max_minutes: float = 10.0
    completed_mean_minutes: float = 4.0

    # Seconds for canceled attempts
    canceled_min_seconds: float = 1.0
    canceled_max_seconds: float = 600.0  # 10 minutes

    def sample_duration(self, completed: bool) -> timedelta:
        if completed:
            # Use triangular distribution around mean within [min, max]
            minutes = random.triangular(
                self.completed_min_minutes,
                self.completed_max_minutes,
                self.completed_mean_minutes,
            )
            return timedelta(minutes=minutes)
        else:
            seconds = random.uniform(self.canceled_min_seconds, self.canceled_max_seconds)
            return timedelta(seconds=seconds)


@dataclass
class SimulationConfig:
    num_users: int = 10
    max_level: int = 100
    start_date: datetime = datetime(2026, 1, 1)

    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    progression: ProgressionConfig = field(default_factory=ProgressionConfig)
    session_duration: SessionDurationConfig = field(default_factory=SessionDurationConfig)

    # Used to label AB-test groups or samples
    sample_id: str = "control"

# ----------------------------
# Helpers
# ----------------------------

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

GENDERS = ["male", "female"]

def random_user_attributes() -> Dict[str, Any]:
    return {
        "location": random.choice(US_STATES),
        "age": random.randint(12, 18),
        "gender": random.choice(GENDERS),
    }

def choose_device(prev_device: Optional[str], cfg: DeviceConfig) -> str:
    if prev_device is None:
        return random.choice(cfg.device_types)

    if random.random() < cfg.stick_with_prev_prob:
        return prev_device
    else:
        # choose among other devices
        others = [d for d in cfg.device_types if d != prev_device]
        return random.choice(others)

def choose_completion(cfg: CompletionConfig) -> bool:
    """Return True if completed, False if canceled."""
    return random.random() < cfg.complete_prob

def choose_pass(level: int,
                max_level: int,
                last_result: Optional[bool],
                difficulty_cfg: DifficultyConfig) -> bool:
    """
    last_result: True if last attempt passed, False if failed, None if no previous.
    """
    probs = difficulty_cfg.adjusted_probs(level, max_level)
    if last_result is True:
        p_pass = probs["after_pass"]
    elif last_result is False:
        p_pass = probs["after_fail"]
    else:  # first attempt at level: use after_pass as a baseline
        p_pass = probs["after_pass"]

    return random.random() < p_pass

def gap_to_next_level(cfg: ProgressionConfig) -> timedelta:
    """
    If the user does NOT go to the next level immediately, choose a gap.
    """
    r = random.random()
    if r < cfg.next_day_prob:
        days = 1
    elif r < cfg.next_day_prob + cfg.next_week_prob:
        days = random.randint(2, 7)  # within next week
    else:
        days = random.randint(8, 30)  # within next month
    return timedelta(days=days)

# ----------------------------
# Core simulation
# ----------------------------

def simulate_user_events(user_id: int, cfg: SimulationConfig) -> List[Dict[str, Any]]:
    events = []

    # --- user_created event ---
    # Everyone is created on start_date; first play happens at a random time that day.
    creation_ts = cfg.start_date
    user_attrs = random_user_attributes()
    user_attrs["sample_id"] = cfg.sample_id

    events.append({
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "timestamp": creation_ts.isoformat(),
        "event_type": "user_created",
        "payload": json.dumps(user_attrs),
    })

    # Random first-play time on that day
    first_play_offset_seconds = random.randint(0, 24 * 60 * 60 - 1)
    current_ts = cfg.start_date + timedelta(seconds=first_play_offset_seconds)

    current_level = 1
    last_pass_result_for_level: Optional[bool] = None
    prev_device: Optional[str] = None

    while current_level <= cfg.max_level:
        # --- play_attempt event ---
        device = choose_device(prev_device, cfg.device)
        completed = choose_completion(cfg.completion)
        level_attempt_id = str(uuid.uuid4())

        # pass/fail only defined for completed attempts
        if completed:
            passed = choose_pass(
                level=current_level,
                max_level=cfg.max_level,
                last_result=last_pass_result_for_level,
                difficulty_cfg=cfg.difficulty,
            )
        else:
            passed = False  # or None; but False keeps it boolean

        session_duration = cfg.session_duration.sample_duration(completed=completed)
        end_ts = current_ts + session_duration

        payload = {
            "level_attempt_id": level_attempt_id,
            "level": current_level,
            "device": device,
            "completed": completed,
            "passed": passed,
            "session_duration_seconds": int(session_duration.total_seconds()),
            "sample_id": cfg.sample_id,
        }

        events.append({
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": current_ts.isoformat(),
            "event_type": "play_attempt",
            "payload": json.dumps(payload),
        })

        # Update "last_pass_result_for_level" only if completed
        if completed:
            last_pass_result_for_level = passed

        # Determine whether to move to next level:
        # If canceled, force repeat same level.
        if completed and passed:
            # Completed and passed -> candidate for moving on
            if random.random() < cfg.progression.immediate_next_level_prob:
                # immediate next level, starting right at end_ts
                current_level += 1
                current_ts = end_ts
            else:
                # schedule after a gap
                gap = gap_to_next_level(cfg.progression)
                current_level += 1
                current_ts = end_ts + gap
        else:
            # failed or canceled -> repeat same level immediately after end_ts
            current_ts = end_ts

        prev_device = device

    return events

def simulate_all_users(cfg: SimulationConfig) -> List[Dict[str, Any]]:
    all_events = []

    # Generate stable UUIDs for each simulated user
    user_ids = [str(uuid.uuid4()) for _ in range(cfg.num_users)]

    for user_id in user_ids:
        user_events = simulate_user_events(user_id, cfg)
        all_events.extend(user_events)
    # Sort all events by timestamp then user_id (just to keep output tidy)
    all_events.sort(key=lambda e: (e["user_id"], e["timestamp"]))
    return all_events

# ----------------------------
# Script entry point
# ----------------------------

if __name__ == "__main__":
    config = SimulationConfig(
        num_users=5,
        max_level=100,
        sample_id="dev_1",
        # You can override any nested config here, e.g.:
        # difficulty=DifficultyConfig(
        #     base_pass_prob_after_pass=0.85,
        #     base_pass_prob_after_fail=0.65,
        #     difficulty_drop_over_game=0.30,
        # ),
    )

    events = simulate_all_users(config)
    output_file = "data/raw/events.csv"

    # Determine whether we need to write the header
    file_exists = os.path.isfile(output_file)
    write_header = not file_exists or os.path.getsize(output_file) == 0

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only if creating a new file
        if write_header:
            writer.writerow(["event_id", "user_id", "timestamp", "event_type", "payload"])

        for e in events:
            writer.writerow([
                e["event_id"],
                e["user_id"],
                e["timestamp"],
                e["event_type"],
                e["payload"],
            ])

    print(f"✔ Wrote {len(events)} events (appended) → {output_file}")