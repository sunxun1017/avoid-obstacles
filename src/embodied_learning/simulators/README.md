# Simulator adapters

The adapters wrap an already-created native vector environment. Environment
creation stays with each simulator's task/config system so launch order,
rendering, device selection, and application lifecycle are not hidden here.

Both adapters expose `reset() -> EnvReset` and
`step(actions) -> EnvStep`. Observation and action objects are passed through
unchanged, preserving Torch tensors, TensorDicts, nested dictionaries, and GPU
placement. `terminated` and `truncated` remain separate so learning code can
handle time limits correctly.

```python
from embodied_learning.simulators import IsaacLabEnvAdapter, MjlabEnvAdapter

# Create `native_env` with that backend's normal task/config workflow first.
env = IsaacLabEnvAdapter(native_env)  # or MjlabEnvAdapter(native_env)
initial = env.reset(seed=123)
transition = env.step(actions)
```

## Backend behavior to account for

- **Isaac Lab**: manager-based and direct environments use vector batches and
  perform same-step auto-reset for completed environments. Manager-based reset
  supports selected environment IDs; direct environments may only expose a
  full reset depending on version/workflow. Camera data may be stale after
  reset unless the task enables rerender-on-reset.
- **mjlab**: manager-based environments use Torch observations/actions and
  support partial reset by environment IDs. The task config controls
  `auto_reset`; with it disabled, the rollout collector owns partial resets.
- Neither adapter converts actor/critic observation group layouts. Tasks should
  expose named observations consistently where cross-backend experiments need
  the same policy inputs.
- Sensor acquisition is not normalized here. Each backend task should provide
  the desired camera/ray sensor output as observations; shared transforms such
  as ADAPT projections belong under `perception/`.

## Installation and lifecycle

Install Isaac Lab according to its official Isaac Sim/version matrix and start
its application before constructing tasks. Install mjlab using its official
package workflow. These simulator stacks are intentionally not unconditional
project dependencies; mixing their Torch/CUDA/runtime pins in one environment
can be fragile. For reliable use, develop against one backend environment at a
time and keep adapter imports lazy at application startup.
