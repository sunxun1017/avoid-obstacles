import torch

from embodied_learning.tasks.cat_stop_transition import CatStopTransitionGait


def state(n=2):
    st = CatStopTransitionGait(n, "cpu", command_deadband=0.01)
    st.reset(
        torch.arange(n),
        torch.ones(n) * 1.4,
        torch.tensor([[0.0, torch.pi]]).repeat(n, 1),
    )
    st.seed(
        torch.tensor([[0.0, 0.2, 0.0, 0.0]]).repeat(n, 1),
        torch.ones(n) * 0.3,
        torch.arange(n),
    )
    return st


def filt(st, dist, speed=0.0, gyro=0.0):
    n = len(st.step)
    cmd = torch.tensor([[0.75, 0.1, 0.0, 0.0]]).repeat(n, 1)
    root = torch.zeros(n, 3)
    goal = torch.zeros(n, 3)
    goal[:, 0] = dist
    v = torch.zeros(n, 3)
    v[:, 0] = speed
    w = torch.zeros(n, 3)
    w[:, 0] = gyro
    return st.filter_command(cmd, root, goal, v, w)


def test_latch_uses_8cm_entry_and_10cm_release():
    s = state()
    assert filt(s, 0.079).count_nonzero() == 0
    assert filt(s, 0.099).count_nonzero() == 0
    assert filt(s, 0.1001)[:, 1].gt(0).all()


def test_emergency_does_not_starve_recovery_steps():
    s = state()
    filt(s, 0.079)
    assert filt(s, 0.09, speed=0.6)[:, 1].gt(0).all()
    out = s.advance(filt(s, 0.09, speed=0.6))
    assert out[:, 0].eq(1).all()


def test_normal_stop_and_restart_features_are_bounded_and_phase_clock_kept():
    s = state()
    s.step += 1
    prev = s.phase_features.clone()
    vec = s.walk_vector.clone()
    for _ in range(12):
        s.advance(torch.zeros(2, 4))
        assert (s.phase_features - prev).abs().max() < 0.5
        assert (s.walk_vector - vec).norm(dim=-1).max() <= 0.10001
        prev = s.phase_features.clone()
        vec = s.walk_vector.clone()
    assert s.activation.eq(0).all() and s.walk_vector.abs().max() < 1e-6
    torch.testing.assert_close(
        s.phase_features, torch.tensor([[1.0, 1.0, 0.0, 0.0]]).repeat(2, 1)
    )
    old = s.phase.clone()
    s.advance(filt(s, 0.11))
    expected = (
        torch.remainder(old + s.phase_dt[:, None] + torch.pi, 2 * torch.pi)
        - torch.pi
    )
    torch.testing.assert_close(s.phase, expected)
    assert (s.phase_features - prev).abs().max() < 0.5


def test_partial_reset_and_restore_do_not_change_other_slots():
    s = state()
    s.advance(filt(s, 0.07))
    snap = s.snapshot()
    s.advance(filt(s, 0.2))
    s.restore(snap, torch.tensor([1]))
    for k, v in snap.items():
        torch.testing.assert_close(getattr(s, k)[1], v[1])
    other = {k: getattr(s, k)[1].clone() for k in snap}
    s.reset(torch.tensor([0]), torch.tensor([1.4]), torch.tensor([[0.0, torch.pi]]))
    for k, v in other.items():
        torch.testing.assert_close(getattr(s, k)[1], v)


def test_task_refresh_keeps_transition_history_on_same_step_partial_reset():
    import copy

    from test_cat_review_regressions import task_case
    from test_cat_risk_guidance import SPEC

    from embodied_learning.tasks.cat_traversal import CatTaskState

    old, c, p, _ = task_case()
    st = CatTaskState(
        old.manifest,
        old.bank,
        old.allowed,
        16,
        "cpu",
        dict(
            noise_level=0.0,
            arrival_control="root_planar_blend_v2",
            command_deadband_m_s=0.01,
            stop_transition="stop8_resume10_v1",
        ),
        risk_guidance=SPEC,
    )
    # task_case bank ignores query shape; expose a separate base root route.
    from types import SimpleNamespace

    class Base:
        def sample_source(self, positions, scenes, goals):
            gf = torch.zeros_like(positions)
            gf[..., 0] = 0.6
            return SimpleNamespace(guidance=gf)

    st.bank.base = Base()
    st.reset(torch.arange(16))
    st.goals[:, 0] = 0.3
    for step in range(4):
        st.refresh(
            copy.deepcopy(p), c, torch.full((16,), step, dtype=torch.long), step
        )
    before = st.gait.snapshot()
    obs = st.observations().clone()
    st.reset(torch.tensor([0]))
    st.refresh(copy.deepcopy(p), c, torch.full((16,), 3, dtype=torch.long), 3)
    for k, v in before.items():
        torch.testing.assert_close(getattr(st.gait, k)[1:], v[1:])
    torch.testing.assert_close(st.observations()["actor"][1:], obs["actor"][1:])


def test_stop_transfer_is_sha_bound_and_rejects_other_task_changes(tmp_path):
    import copy
    from types import SimpleNamespace

    from test_cat_curriculum import fixture_contract, fixture_metadata

    from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart
    from embodied_learning.tasks.cat_foot_guidance import sha256

    old = fixture_contract(4)
    old["schema"]["task"]["arrival_control"] = "root_planar_blend_v2"
    meta = fixture_metadata(4)
    p = tmp_path / "parent.pt"
    torch.save(
        dict(
            field_contract=old,
            curriculum=meta,
            training_scenes=[2],
            actor_state_dict={"w": torch.ones(2)},
            completed_iterations=280,
        ),
        p,
    )
    new = copy.deepcopy(old)
    new["schema"]["task"]["stop_transition"] = "stop8_resume10_v1"
    target = {
        **meta,
        "experiment": "stop10",
        "protocol_sha256": "stop10",
        "stop_transition_refinement": dict(
            version="cat_stop_transition_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment=meta["experiment"],
            parent_protocol_sha256=meta["protocol_sha256"],
            before=None,
            after="stop8_resume10_v1",
        ),
    }

    def load(schema, md):
        return load_actor_warmstart(
            p,
            contract=SimpleNamespace(to_dict=lambda: schema),
            curriculum=md,
            training_scenes=[2],
            device="cpu",
        )

    _, info = load(new, target)
    assert info["actor_restored"] and not info["optimizer_restored"]
    import pytest

    bad = copy.deepcopy(new)
    bad["schema"]["task"]["push"] = False
    with pytest.raises(ValueError):
        load(bad, target)
    bad = copy.deepcopy(target)
    bad["stop_transition_refinement"]["parent_sha256"] = "bad"
    with pytest.raises(ValueError):
        load(new, bad)
