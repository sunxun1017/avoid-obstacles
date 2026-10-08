"""Official RSL-RL PPO with portable, strict field-policy checkpoints."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from embodied_learning.protocols.imitation import (
    FieldTrainingEnvironment,
    PolicyContract,
)


@dataclass(frozen=True)
class PpoConfig:
    """Optimization settings; iterations means additional updates on resume."""

    iterations: int = 30000
    rollout_steps: int = 24
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] | None = None
    normalization: bool = True
    gamma: float = 0.99
    epochs: int = 5
    mini_batches: int = 4
    learning_rate: float = 0.001
    save_interval: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "iterations",
            "rollout_steps",
            "epochs",
            "mini_batches",
            "save_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not self.hidden_dims or any(
            type(d) is not int or d < 1 for d in self.hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")

        if self.critic_hidden_dims is not None and (
            not self.critic_hidden_dims
            or any(type(d) is not int or d < 1 for d in self.critic_hidden_dims)
        ):
            raise ValueError("Invalid critic_hidden_dims.")
        if (
            type(self.normalization) is not bool
            or not math.isfinite(self.gamma)
            or not 0 < self.gamma <= 1
        ):
            raise ValueError("Invalid normalization/gamma.")

    def runner_config(self) -> dict:
        """RSL-RL 5.4.2 configuration, no simulator dependency or W&B login."""
        model = dict(
            class_name="MLPModel",
            hidden_dims=list(self.hidden_dims),
            activation="elu",
            obs_normalization=self.normalization,
        )
        return dict(
            actor={
                **model,
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                },
            },
            critic={
                **model,
                "hidden_dims": list(self.critic_hidden_dims or self.hidden_dims),
            },
            obs_groups={"actor": ["actor"], "critic": ["critic"]},
            algorithm=dict(
                class_name="PPO",
                value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                entropy_coef=0.01,
                num_learning_epochs=self.epochs,
                num_mini_batches=self.mini_batches,
                learning_rate=self.learning_rate,
                schedule="adaptive",
                gamma=self.gamma,
                lam=0.95,
                desired_kl=0.01,
                max_grad_norm=1.0,
                rnd_cfg=None,
                symmetry_cfg=None,
            ),
            num_steps_per_env=self.rollout_steps,
            save_interval=self.save_interval,
            logger="tensorboard",
            seed=self.seed,
            check_for_nan=True,
        )


def train_specialist(
    env: FieldTrainingEnvironment,
    config: PpoConfig,
    output_dir: Path,
    *,
    resume: Path | None = None,
) -> Path:
    """Train or strictly resume an official PPO specialist and return latest.pt.

    Torch checkpoints contain Python training state: load trusted files only.
    The last checkpoint and each periodic checkpoint embed their full contract.
    """
    if env.num_envs * config.rollout_steps < config.mini_batches:
        raise ValueError("Rollout smaller than the number of PPO mini-batches.")
    contract = PolicyContract.from_env(
        env,
        config.hidden_dims,
        critic_hidden_dims=config.critic_hidden_dims,
        normalization=config.normalization,
    )
    training_scenes = sorted(set(env.scene_ids.tolist()))
    # Native scene allow-list is stable even when not every scene is sampled.
    if hasattr(env, "field_state"):
        training_scenes = list(env.field_state.allowed)
    restored = None
    if resume:
        restored = torch.load(resume, map_location=env.device, weights_only=False)
        contract.require(PolicyContract.from_dict(restored["field_contract"]))
        expected, actual = asdict(config), dict(restored["ppo_config"])
        for key in ("iterations", "save_interval"):
            expected.pop(key)
            actual.pop(key)
        actual["hidden_dims"] = tuple(actual["hidden_dims"])
        actual.setdefault("critic_hidden_dims", None)
        actual.setdefault("normalization", True)
        actual.setdefault("gamma", 0.99)
        if actual["critic_hidden_dims"] is not None:
            actual["critic_hidden_dims"] = tuple(actual["critic_hidden_dims"])
        if expected != actual or restored["training_scenes"] != training_scenes:
            raise ValueError("PPO training configuration/scene contract mismatch.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = output_dir / "config.json"
    serial_config = json.loads(json.dumps(asdict(config)))
    if configuration_path.exists():
        old = json.loads(configuration_path.read_text())
        old.setdefault("critic_hidden_dims", None)
        old.setdefault("normalization", True)
        old.setdefault("gamma", 0.99)
        comparable_old, comparable_new = old.copy(), serial_config.copy()
        for key in ("iterations", "save_interval"):
            comparable_old.pop(key)
            comparable_new.pop(key)
        if comparable_old != comparable_new:
            raise ValueError("Output directory already has another configuration.")
    else:
        configuration_path.write_text(json.dumps(serial_config, indent=2) + "\n")
    torch.manual_seed(config.seed)
    initial, losses = [], {}

    class FieldRunner(OnPolicyRunner):
        def save(self, path: str, infos: dict | None = None) -> None:
            state = self.alg.save()
            state.update(
                iter=self.current_learning_iteration,
                completed_iterations=self.current_learning_iteration + 1,
                environment_steps=int(getattr(env, "training_step", 0)),
                infos=infos,
                field_contract=contract.to_dict(),
                ppo_config=asdict(config),
                training_scenes=training_scenes,
                last_losses=losses.copy(),
                parameter_delta=float(
                    (
                        torch.nn.utils.parameters_to_vector(
                            self.alg.actor.parameters()
                        ).detach()
                        - initial[0]
                    ).norm()
                ),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            )
            temporary = Path(path).with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)

    runner = FieldRunner(
        env,
        copy.deepcopy(config.runner_config()),
        log_dir=str(output_dir),
        device=str(env.device),
    )
    if restored:
        runner.alg.load(restored, load_cfg=None, strict=True)
        runner.current_learning_iteration = restored["completed_iterations"]
        if hasattr(env, "training_step"):
            env.training_step = restored.get("environment_steps", 0)
        torch.set_rng_state(restored["torch_rng"].cpu())
        if restored["cuda_rng"]:
            torch.cuda.set_rng_state_all([r.cpu() for r in restored["cuda_rng"]])
    initial.append(
        torch.nn.utils.parameters_to_vector(runner.alg.actor.parameters())
        .detach()
        .clone()
    )
    official_update = runner.alg.update

    def checked_update():
        result = official_update()
        if not all(math.isfinite(v) for v in result.values()):
            raise RuntimeError("PPO produced nonfinite optimization losses.")
        losses.update(result)
        with (output_dir / "losses.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        return result

    runner.alg.update = checked_update
    runner.learn(config.iterations)
    path = output_dir / "latest.pt"
    runner.save(str(path))
    return path
