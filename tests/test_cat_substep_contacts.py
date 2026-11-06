from types import SimpleNamespace

import torch

from embodied_learning.evaluation.cat_substep_contacts import capture_prohibited_contacts


def test_contact_at_first_substep_survives_last_clear_sample_and_resets_next_step():
    found = torch.zeros((2, 1), dtype=torch.bool)
    obstacle = SimpleNamespace(data=SimpleNamespace(found=found))
    clear = SimpleNamespace(data=SimpleNamespace(found=torch.zeros_like(found)))
    scene = type("Scene", (), {
        "__getitem__": lambda self, name: obstacle if name == "bar" else clear,
        "update": lambda self: None,
    })()
    sim = SimpleNamespace(sense=lambda: None)
    native = SimpleNamespace(scene=scene, sim=sim, cat_obstacle_contacts=["bar"])
    original_outcomes = {"prohibited_contact": torch.tensor([False, True])}

    def step(first_hit):
        found[:, 0] = torch.tensor([first_hit, False])
        scene.update()
        found.zero_()
        scene.update()
        return None, None, None, {"field_outcomes": original_outcomes}

    env = SimpleNamespace(unwrapped=native, device="cpu", num_envs=2, step=step)
    with capture_prohibited_contacts(env):
        assert env.step(True)[3]["field_outcomes"]["prohibited_contact"].tolist() == [True, True]
        assert env.step(False)[3]["field_outcomes"]["prohibited_contact"].tolist() == [False, True]
    assert env.step is step
    assert original_outcomes["prohibited_contact"].tolist() == [False, True]
