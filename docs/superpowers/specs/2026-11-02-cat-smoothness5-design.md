# Both smoothness penalties ×5, latest difficulty,600updates

User explicitly requests action and joint smoothness costs ×5 and600updates at latest difficulty. This supersedes the unlaunched action-only3×/200 proposal.

Use stage4 original baseline128 SHA22505bcb518ef36eb3b064e3ec7e7eed2f510c02154bec098a6e769ef3903e1f. Preserve all prior bugfixes and source reward formulas; action -.001→-.005, joint -1e-6→-5e-6. No landing/contact-force/fullbody-velocity pilot penalties. Stage4:12cm bar,24cm depth, overhead1.20m, gap.70m. PD/noise/randomization/fields unchanged.

Explicit same-stage Actor transfer changes only two declared reward schema weights, checking exact parent SHA/protocol and physics. The saved Actor is inherited; Critic/Adam restart because reward changes. Subsequent40update chunks strictly restore full PPO state. No default weight changes; no auto-promotion. Frozen original baseline remains selected until evaluation warrants a user decision.

Run600updates on a nonzero4090. Every40updates run256 source_train episodes seed20261102 and32 passive500Hz motion/force episodes seed20261026; final256 independent seed20261103. Keep all checkpoints, no early promotion from a lucky gate. Record success, contacts, arrival, goalhold, postbar rootomega, forcepeaks and joint HF RMS. No claims reward cures jitter without evaluations. All existing noise and RFI remain enabled.
