import pytest

from embodied_learning.simulators import (
    IsaacLabEnvAdapter,
    MjlabEnvAdapter
)

class FakeEnv:
    num_envs = 2
    device = "gpu"

    def reset(self, **kwargs):
        return {"policy" : [[0.0],[0.0]]},{}

    def step(self, actions):
        return (
            {"policy" : [[1.0], [1.0]]},
            [0.0, 0.0],
            [False, False],
            [False, False],
            {},
        )

    def close(self):
        self.closed = True
@pytest.mark.parametrize("adapter_type", [IsaacLabEnvAdapter, MjlabEnvAdapter])
def test_adapter_interface(adapter_type):
    # for adapter_type in (IsaacLabEnvAdapter, MjlabEnvAdapter):
    native = FakeEnv()  # 也就是把这个环境塞进去
    env = adapter_type(native)

    reset = env.reset(seed = 0)
    step = env.step([[0.0], [0.0]])

    assert env.num_envs == 2
    assert reset.observations["policy"][0] == [0.0]
    assert step.observations["policy"][0] == [1.0]
    assert step.terminated[0] == False # 为什么一个元组会变成这样，关键是因为适配了一下
    assert step.truncated == [False, False]

    env.close()
    assert native.closed

def test_register_gpu_status():
    import torch
    import mjlab.tasks
    from mjlab.tasks.registry import list_tasks

    task_id = "Mjlab-Velocity-Flat-Unitree-G1"

    assert task_id in list_tasks()
    assert torch.cuda.is_available()

def test_mjlab_g1_gpu_env_smoke():
    import torch
    import mjlab.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping test.")

    task_id = "Mjlab-Velocity-Flat-Unitree-G1"
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 2

    native = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env = MjlabEnvAdapter(native)   # 把这个环境放进去

    try:
        reset = env.reset(seed=0)

        action_dim = native.action_manager.total_action_dim
        actions = torch.zeros(
            (env.num_envs, action_dim),
            device="cuda:0",
        )
        step = env.step(actions)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete

        assert env.num_envs == 2
        assert reset.observations
        assert step.observations
        assert step.rewards.shape[0] == 2
        assert step.rewards.shape[0] == 2
        assert step.terminated.shape[0] == 2
        assert step.truncated.shape[0] == 2
    finally:
        env.close()