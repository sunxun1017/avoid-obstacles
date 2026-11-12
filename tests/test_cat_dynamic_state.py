from types import SimpleNamespace

import torch


def joint_case():
    from test_cat_review_regressions import task_case
    from embodied_learning.tasks.cat_dynamic_state import JointTaskState
    base, controller, payload, _ = task_case()
    manifest = SimpleNamespace(scenes=[SimpleNamespace(id='flat', boxes=[],
        starts=[(0., 0., .8)], goals=[(2., 0., .8)])])
    base.bank.sample_source = lambda p, *args: SimpleNamespace(
        guidance=torch.zeros_like(p), sdf_gradient=torch.zeros_like(p),
        sdf=torch.ones(p.shape[:2]))
    state = JointTaskState(manifest, base.bank, (0,), 16, 'cpu',
        dict(noise_level=0., traversal_robustness='obstacle_clearance_v1',
             posture_reward='overhead_clearance_v1'), joint_config={'gait_scale': 1., 'stage': 1})
    state.reset(torch.arange(16))
    state.goals[:, 0] = 2.
    centers = torch.zeros(16, 2, 3)
    centers[:, :, 2] = 1.
    payload['foot_cover'] = (centers, torch.tensor([.05, .05]), torch.tensor([0, 1]))
    payload['joint_geometry'] = dict(centers=centers, radii=torch.tensor([.05, .05]),
        upright_head=centers[:, :1].clone(), head_radii=torch.tensor([.05]))
    payload['dynamic'] = dict(position=torch.ones(16, 3)*10, velocity=torch.zeros(16, 3),
        observed_position=torch.ones(16, 3)*10, observed_velocity=torch.zeros(16, 3),
        radius=torch.ones(16)*.08, valid=torch.zeros(16, dtype=torch.bool),
        episode_dynamic=torch.zeros(16, dtype=torch.bool), age=torch.zeros(16))
    return state, controller, payload


def test_joint_state_appends_features_without_reinterpreting_action_history():
    state, c, p = joint_case()
    state.refresh(p, c, torch.zeros(16, dtype=torch.long), 0)
    obs = state.observations()
    assert obs['actor'].shape == (16, 191)
    assert obs['critic'].shape == (16, 279)
    assert obs['anchor'].shape == (16, 163)
    torch.testing.assert_close(obs['actor'][:, 52:64], c.actions)
    assert not obs['actor'][:, 172:181].any()  # absent ball must have no leaked location


def test_joint_partial_reset_preserves_terminal_report_and_other_observations():
    state, c, p = joint_case()
    state.refresh(p, c, torch.zeros(16, dtype=torch.long), 0)
    p['physical_collision'][0] = True
    state.refresh(p, c, torch.ones(16, dtype=torch.long), 1)
    obs = state.observations().clone()
    assert state.reward()[0] <= -25
    assert state.saved_outcomes['collision'][0]
    state.reset(torch.tensor([0]))
    p['physical_collision'][0] = False
    state.refresh(p, c, torch.ones(16, dtype=torch.long), 1)
    torch.testing.assert_close(state.observations()['actor'][1:], obs['actor'][1:])
    assert state.saved_outcomes['collision'][0]
    state.refresh(p, c, torch.ones(16, dtype=torch.long)*2, 2)
    assert not state.saved_outcomes['collision'][0]


def test_joint_finite_horizon_timeout_has_negative_terminal_reward():
    state, c, p = joint_case()
    state.refresh(p, c, torch.zeros(16, dtype=torch.long), 0)
    state.refresh(p, c, torch.ones(16, dtype=torch.long)*1000, 1)
    assert state.outcomes()['timeout'].all()
    assert (state.reward() <= -10).all()
    assert (state.reward() > -11).all()


def test_joint_corridor_does_not_force_back_to_center():
    from embodied_learning.tasks.cat_dynamic_state import JointTraversalSafety
    manifest = SimpleNamespace(scenes=[SimpleNamespace(boxes=[SimpleNamespace(
        center=(0., 0., .05), half_size=(.1, .9, .05))])])
    safety = JointTraversalSafety(manifest, 'cpu')
    cmd = torch.tensor([[.75, .3, .2, 0.]])
    actual, _ = safety.corridor(cmd, torch.tensor([[0., .4, .8]]), torch.tensor([0]))
    torch.testing.assert_close(actual, cmd)


def test_body_prediction_stops_extrapolating_after_motion_stops():
    state, c, p = joint_case()
    p['dynamic']['episode_dynamic'].fill_(True)
    p['dynamic']['position'][:] = torch.tensor([1.1, 0., 1.])
    state.refresh(p, c, torch.zeros(16, dtype=torch.long), 0)
    p['joint_geometry']['centers'][:, :, 0] = .1
    state.refresh(p, c, torch.ones(16, dtype=torch.long), 1)
    state.refresh(p, c, torch.ones(16, dtype=torch.long)*2, 2)
    torch.testing.assert_close(state.saved_reward_diagnostics['joint_clearance'], torch.full((16,), .87))
