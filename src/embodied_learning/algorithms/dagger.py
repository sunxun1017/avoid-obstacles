"""Online student-state DAgger through simulator-independent protocols."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tensordict import TensorDict

from embodied_learning.data.aggregation import AggregationBuffer
from embodied_learning.protocols.imitation import (
    ExpertPolicy,
    ImitationEnvironment,
    PolicyContract,
    StudentPolicy,
)


@dataclass(frozen=True)
class DaggerConfig:
    """Additional aggregation rounds; linear expert mixing reaches beta zero."""

    iterations: int = 1000
    rollout_steps: int = 64
    updates: int = 100
    batch_size: int = 512
    capacity: int = 200000
    learning_rate: float = 0.0003
    beta_start: float = 1.0
    beta_rounds: int = 20
    lr_decay: float = 0.999
    save_interval: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "iterations",
            "rollout_steps",
            "updates",
            "batch_size",
            "capacity",
            "beta_rounds",
            "save_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("learning_rate", "beta_start", "lr_decay"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.learning_rate <= 0 or not 0 <= self.beta_start <= 1:
            raise ValueError("Invalid learning rate or beta_start.")
        if not 0 < self.lr_decay <= 1:
            raise ValueError("lr_decay must be in (0,1].")

    def beta(self, round_index: int) -> float:
        """Explicit mixture probability; zero from beta_rounds onward."""
        return self.beta_start * max(0.0, 1.0 - round_index / self.beta_rounds)


def train_dagger(
    env: ImitationEnvironment,
    student: StudentPolicy,
    experts: Mapping[int, ExpertPolicy],
    config: DaggerConfig,
    output_dir: Path,
    *,
    resume: Path | None = None,
) -> Path:
    """Label current student observations, aggregate, fit and persist all state.

    The teacher receives its complete original TensorDict before env.step.
    Native actor/critic groups need not be identical. Replay stores only the
    student actor group; expert privileged observations are never replayed.
    A resumed run starts fresh physical episodes, retaining learner/data/RNG.
    """
    if getattr(env, "inference_only", False):
        raise ValueError("Cannot train an inference-only environment.")
    contract = student.contract
    contract.require(
        PolicyContract.from_env(env, contract.hidden_dims), network=False
    )
    allowed = set(
        getattr(getattr(env, "field_state", None), "allowed", env.scene_ids.tolist())
    )
    if not allowed.issubset(experts):
        raise ValueError("Missing teacher for one or more training scenes.")
    teachers = {}
    for scene_id in sorted(allowed):
        expert = experts[scene_id]
        contract.require(expert.contract, network=False)
        if (
            hasattr(expert, "training_scenes")
            and scene_id not in expert.training_scenes
        ):
            raise ValueError("Teacher scene contract does not match routing.")
        teachers[scene_id] = dict(
            contract=expert.contract.to_dict(),
            identity=getattr(expert, "identity", None),
        )
    buffer = AggregationBuffer(
        config.capacity, contract.actor_dim, contract.action_dim, "cpu"
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    completed = 0
    torch.manual_seed(config.seed)
    if resume:
        saved = torch.load(resume, map_location=env.device, weights_only=False)
        contract.require(PolicyContract.from_dict(saved["field_contract"]))
        expected, actual = asdict(config), dict(saved["dagger_config"])
        for key in ("iterations", "save_interval"):
            expected.pop(key)
            actual.pop(key)
        if expected != actual or teachers != saved["teachers"]:
            raise ValueError("DAgger configuration/teacher contract mismatch.")
        student.load_state_dict(saved["student"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        buffer.load_state_dict(saved["buffer"])
        completed = saved["completed_rounds"]
        if hasattr(env, "training_step"):
            env.training_step = saved.get("environment_steps", 0)
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = env.get_observations()
    student.train()

    def save(path: Path, round_count: int) -> None:
        state = dict(
            field_contract=contract.to_dict(),
            student=student.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            buffer=buffer.state_dict(),
            completed_rounds=round_count,
            environment_steps=int(getattr(env, "training_step", 0)),
            dagger_config=asdict(config),
            teachers=teachers,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        )
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    for round_index in range(completed, completed + config.iterations):
        beta = config.beta(round_index)
        with torch.no_grad():
            for _ in range(config.rollout_steps):
                # Clone before stepping an environment that may mutate its buffers.
                snapshot = obs.clone()
                scenes = env.scene_ids.clone()
                labels = torch.zeros(
                    env.num_envs, env.num_actions, device=env.device
                )
                for scene_id in allowed:
                    mask = scenes == scene_id
                    if mask.any():
                        labels[mask] = experts[scene_id].predict(snapshot[mask])
                student.update_normalization(snapshot)
                predicted = student.predict(snapshot)
                if (
                    predicted.shape != labels.shape
                    or not torch.isfinite(predicted).all()
                ):
                    raise ValueError("Student action shape/value mismatch.")
                buffer.add(snapshot["actor"], labels)
                choose_teacher = (
                    torch.rand(env.num_envs, 1, device=env.device) < beta
                )
                actions = torch.where(choose_teacher, labels, predicted)
                obs, _, _, _ = env.step(actions)
        loss_value = 0.0
        for _ in range(config.updates):
            features, targets = buffer.sample(config.batch_size)
            training_obs = TensorDict(
                {"actor": features.to(env.device)}, batch_size=[len(features)]
            )
            prediction = student(training_obs)
            loss = (prediction - targets.to(env.device)).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("DAgger produced nonfinite supervised loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            loss_value += loss.item() / config.updates
        scheduler.step()
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        round=round_index + 1,
                        beta=beta,
                        samples=len(buffer),
                        seen=buffer.seen,
                        mse=loss_value,
                    )
                )
                + "\n"
            )
        if (round_index + 1) % config.save_interval == 0:
            save(output_dir / f"round_{round_index + 1}.pt", round_index + 1)
        save(output_dir / "latest.pt", round_index + 1)
    return output_dir / "latest.pt"


@torch.no_grad()
def evaluate_student(
    env: ImitationEnvironment, student: ExpertPolicy, *, steps: int
) -> dict:
    """Evaluate only the supplied policy; no teacher or mixing path exists."""
    if steps < 1:
        raise ValueError("Evaluation steps must be positive.")
    obs = env.get_observations()
    reward_sum, episodes = 0.0, 0
    counts: dict[str, int] = {}
    final_distance, min_distance, stopped, metric_steps = 0.0, 0.0, 0, 0
    final_count, min_count = 0, 0
    for _ in range(steps):
        obs, rewards, dones, extras = env.step(student.predict(obs))
        reward_sum += float(rewards.sum())
        episodes += int(dones.sum())
        for key, flag in extras.get("field_outcomes", {}).items():
            counts[key] = counts.get(key, 0) + int((flag & dones.bool()).sum())
        metrics = extras.get("cat_metrics", {})
        if metrics:
            final = metrics["goal_distance"][dones.bool()]
            minimum = metrics["min_goal_distance"][dones.bool()]
            final_distance += float(final[torch.isfinite(final)].sum())
            min_distance += float(minimum[torch.isfinite(minimum)].sum())
            final_count += int(torch.isfinite(final).sum())
            min_count += int(torch.isfinite(minimum).sum())
            stopped += int(metrics["stopped"].sum())
            metric_steps += env.num_envs
    return dict(
        mean_final_goal_distance=final_distance / final_count
        if final_count
        else None,
        mean_min_goal_distance=min_distance / min_count if min_count else None,
        stopped_step_fraction=stopped / metric_steps if metric_steps else None,
        steps=steps,
        envs=env.num_envs,
        episodes=episodes,
        mean_step_reward=reward_sum / (steps * env.num_envs),
        outcomes=counts,
        rates={k: v / episodes if episodes else None for k, v in counts.items()},
    )
