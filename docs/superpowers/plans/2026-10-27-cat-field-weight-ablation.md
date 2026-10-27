# CAT stage4 field-reward comparison

Goal: diagnose physical obstacle contacts, then compare unchanged reward, GF-only and DF-only 1.5x weights from the same stage4 update480 Actor.

Approved scope: user's acceptance of the preceding single-factor comparison. Keep current task profile, stage4 geometry, PD, noise, pushes, RFI, foot-progress gate, optimizer configuration and all historical outputs. Use nonzero physical GPUs only.

1. Instrument frozen-parent inference at 256 environments, source_train seed20261026. Read actual contact pairs before automatic reset; attribute beam/bar, body part, point SDF, GF alignment, root progress and first-contact time. Assert aggregate outcomes reproduce previous 233/256 clean baseline.
2. Add independent field-term weights to the common runtime, rather than changing coupled ground/overhead masks. Add explicitly declared, parent-hash checked Actor-only reward transfer within the same geometry/stage; do not relax unrelated contract checks. Baseline uses the same transfer path and fresh Critic/optimizer.
3. Regression tests: term independence, invalid weights, transfer parent/declaration checks, unchanged geometry/PD/RFI/gate requirements, strict rejection of full resume into altered weights. Red then green; full CPU suite; GPU smoke on each arm.
4. Select affected head/feet terms from diagnostic contacts. Trial budget: 128 updates per arm, 4096 envs, 160 controls/update (83886080 new transitions per arm). Evaluate256 at updates32,64,96,128 with matched seed20261026; final independent-seed20261027 and final no-push evaluations. This is a pilot; do not auto-promote a winning arm or claim statistically established gains.
5. Start controlled trials on physical GPU6, sequentially if other nonzero4090 cards lack headroom. Verify common parent Actor exact, fresh Critic/optimizer, actual weights and physical GPU UUID, finite optimizer losses and first formal update. Save launch/status/results locally.

Ruling: include a continued-training baseline, since comparing modified arms only to the frozen checkpoint confounds weight effects with further PPO training.
Ruling: 1.5x and128updates are bounded pilot choices; preserve original611update models. Select checkpoint on training-evaluation seed only; report independent-seed results separately.


Diagnostic findings: source_train seed20261026 yielded232/256 clean episodes
(24 contact episodes, all ground bar;19right foot and5left foot). Historical
same-seed evaluation was233/256. Record actual results rather than require GPU
bitwise replay. Query-point SDF at first contact remained0.071–0.102m, while
native contact distances were negative. First-contact velocity is post-impact;
its GF cosine alone does not establish the pre-impact cause.

Ruling: adjust only feetgf or feetdf, since all observed first contacts involve
feet and no first contact involves the beam. Keep head/hands/knees/shoulder
coefficients and task masks unchanged.

Execution ledger: independent runtime override and parent-checked transfer
implemented; regression tests observed red then green; explicit full-PPO resume
rejection tested; review found no blocking correctness issues. Native smoke and
formal launch records are stored under the existing remote experiment in
stage4-field-reward-trials-20261027, without a per-arm Python source copy.
