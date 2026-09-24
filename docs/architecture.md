# Architecture

## Core boundaries

- **Protocols** define contracts for environments, policies, experts, and data.
- **Algorithms** orchestrate rollouts and optimization through those contracts.
- **Data** owns trajectory schemas, dataset aggregation, and persistence.
- **Policies** own inference and model components, not environment stepping.
- **Perception** maps sensor measurements to observations; it does not know
  which simulator produced them.
- **Simulator adapters** translate simulator APIs into project contracts.
- **Robots** define embodiment-specific state/action conventions.
- **Tasks** compose robot, environment, reward, and observation choices.
- **Evaluation** measures behavior without changing training semantics.
- **CLI/configs** assemble implementations at the application boundary.

## Dependency rules

1. `protocols` must not import any other project package.
2. `algorithms` may depend on `protocols`, `data`, and policy interfaces. It
   must not import a simulator, robot model, task implementation, or perception
   backend directly.
3. `data`, `policies`, and `perception` depend only on protocols and ordinary
   numerical/modeling libraries.
4. `simulators`, `robots`, and `tasks` implement or compose protocols. They do
   not become dependencies of generic algorithms.
5. `cli` is the composition root: it resolves configuration and constructs the
   selected adapters, task, policy, and algorithm.

## DAgger data flow

The environment produces the student's deployable observation and task state.
The student action drives the rollout. An expert receives its own observation
(which may include privileged state) and labels the state visited by the
student. The dataset stores the student observation with that expert label.
The algorithm does not assume that expert and student observation spaces are
identical; only their action contract must be compatible.

```text
task/environment -> student observation -> student policy -> action -> environment
         |                                  |
         +-> expert observation -> expert -+-> expert action label
                                             |
                           (student observation, expert action) -> dataset -> learner
```

## Extension seam for ADAPT

ADAPT belongs under `perception/` as an observation representation that maps a
sensor sample and radius into elevation and distance projections. It should
not depend on DAgger. The policy may consume these projections and optionally
predict a radius; task configuration defines how radius actions affect sensor
processing. Simulator-specific point-cloud acquisition belongs in the
simulator/sensor adapter, with a common sensor sample passed to the projection.

## Third-party integration

Prefer adapting stable external APIs at package boundaries over copying their
internals. If `imitation` or RSL-RL is used, keep the integration in a backend
module and preserve a framework-independent algorithm contract. Record source,
version, and license for borrowed implementations and robot assets.
