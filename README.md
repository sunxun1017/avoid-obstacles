# Embodied Learning

A modular research codebase for embodied imitation learning. Algorithms, robot
tasks, simulators, and perception representations are kept behind small
interfaces so they can be combined without making one depend on another.

## Design goals

- Keep learning algorithms independent of Isaac Lab, mjlab, and Gymnasium APIs.
- Treat sensors and observation representations as replaceable components.
- Keep robot-specific action mappings and task definitions out of algorithm code.
- Make experiments reproducible through explicit, composable configuration.
- Start with a small working vertical slice before adding framework machinery.

## Repository map

```text
configs/                         Experiment configuration examples
docs/                            Architecture and implementation notes
examples/                        Small end-to-end usage examples
src/embodied_learning/
  algorithms/                     DAgger, behavior cloning, and trainers
  cli/                            User-facing commands
  data/                           Trajectory records and aggregated datasets
  evaluation/                     Metrics, rollouts, and reports
  perception/                     Sensor processing and observation encoders
  policies/                       Policy interfaces and reusable networks
  protocols/                      Framework-independent contracts
  robots/                         Robot definitions and action mappings
  simulators/                     Simulator adapters
  tasks/                          Task definitions and task-specific rewards
tests/                            Unit and integration tests
```

The directories are architectural boundaries, not promises that each module is
already implemented. The initial dependency set is intentionally empty; each
simulator or learning backend should be an optional extra when it is added.

## Suggested development order

1. Define framework-independent contracts for observations, actions, experts,
   policies, environments, and trajectory storage.
2. Implement DAgger collection and behavior-cloning updates against those
   contracts, with a tiny Gymnasium-compatible example environment.
3. Add one simulator adapter and one robot/task vertical slice.
4. Add sensor adapters and reusable perception representations, including the
   ADAPT dual-projection representation as one optional implementation.
5. Add evaluation and experiment configuration once the first workflow runs.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

Code formatting, naming, comments, and docstring conventions are documented
in [`CONTRIBUTING.md`](CONTRIBUTING.md). Python style is enforced by Ruff and
EditorConfig (85-character line limit, four-space indentation).

Isaac Lab and mjlab adapters are available under
`embodied_learning.simulators`. They wrap native vector environments without
importing either simulator at package import time. See
[`src/embodied_learning/simulators/README.md`](src/embodied_learning/simulators/README.md)
for setup and behavioral notes. G1 paper-field PPO specialists and online
DAgger training are available; see [the server training guide](docs/field_training.md).

CAT also has a decoupled online deployment path using depth-camera or LiDAR
measurements, without loading offline field caches. See
[CAT online mapping and playback](docs/cat_online.md) for commands, source
correspondence, GPU boundaries, and current limitations.

## License

The project is currently marked MIT. Verify this choice before distributing
code that incorporates third-party implementations or assets.
