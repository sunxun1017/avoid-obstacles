# Click-and-Traverse

`src/embodied_learning/perception/voxel_field.py` adapts the field-construction
algorithm in `procedural_obstacle_generation/pf_modular.py` from
GalaxyGeneralRobotics/Click-and-Traverse, licensed under Apache-2.0.

Source: https://github.com/GalaxyGeneralRobotics/Click-and-Traverse

Reference commit: `866ba392f1c1e84b92ad75fa66550f26e8af8e48`.

The adapted file records modifications in its module docstring. The upstream
license is preserved in `licenses/Click-and-Traverse-LICENSE`. The adaptation
is subject to Apache-2.0; the project's general MIT declaration does not
replace these third-party terms. Paper-formula functions are implemented
separately. Detailed provenance and differences: `docs/potential_fields.md`.

# G1 field training provenance

`robots/g1.py` uses the ordered body-site names and missing-site offsets from
Click-and-Traverse `data/assets/unitree_g1/g1_mjx_feetonly_torque.xml` at the
reference commit above. This is a limited schema adaptation, not a copy of the
author's trainer. The same preserved Apache-2.0 license applies to this adaptation.
`field_observation.py` and `field_reward.py` implement the paper equations
independently; generic DAgger is implemented here.

PPO and actor normalization are provided through public APIs of RSL-RL 5.4.2
(BSD-3-Clause), with mjlab 1.6.0 (Apache-2.0) providing the G1 model, control and
simulation backend. These packages/assets remain external dependencies; their
licenses must be retained when redistributed. Their source is not vendored here.
See `docs/field_training.md` for action/body/reward/backend differences.
