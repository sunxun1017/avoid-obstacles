from types import SimpleNamespace

import torch

from embodied_learning.evaluation.cat_substep_contacts import (
    capture_prohibited_contacts,
)


def test_contact_at_first_substep_survives_last_clear_sample_and_resets_next_step():
    found = torch.zeros((2, 1), dtype=torch.bool)
    obstacle = SimpleNamespace(data=SimpleNamespace(found=found))
    clear = SimpleNamespace(data=SimpleNamespace(found=torch.zeros_like(found)))
    scene = type(
        "Scene",
        (),
        {
            "__getitem__": lambda self, name: obstacle if name == "bar" else clear,
            "update": lambda self: None,
        },
    )()
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
        assert env.step(True)[3]["field_outcomes"][
            "prohibited_contact"
        ].tolist() == [True, True]
        assert env.step(False)[3]["field_outcomes"][
            "prohibited_contact"
        ].tolist() == [False, True]
    assert env.step is step
    assert original_outcomes["prohibited_contact"].tolist() == [False, True]


def test_training_accumulator_exposes_transient_contact_before_reward_and_clears_next_step():
    from embodied_learning.tasks.cat_substep_contacts import SubstepContacts

    hit = torch.zeros((2, 1), dtype=torch.bool)
    ground = torch.ones_like(hit)  # ground is deliberately not a prohibited sensor
    obstacle = SimpleNamespace(data=SimpleNamespace(found=hit))
    clear = SimpleNamespace(data=SimpleNamespace(found=torch.zeros_like(hit)))
    scene = type(
        "Scene",
        (),
        {
            "__getitem__": lambda self, name: obstacle if name == "bar" else clear,
            "update": lambda self: None,
        },
    )()
    native = SimpleNamespace(
        scene=scene,
        sim=SimpleNamespace(sense=lambda: None),
        cat_obstacle_contacts=["bar"],
        num_envs=2,
        device="cpu",
    )
    capture = SubstepContacts(native)
    capture.begin()
    hit[0] = True
    scene.update()
    hit.zero_()
    scene.update()
    assert capture.obstacle.tolist() == [True, False]
    assert capture.prohibited.tolist() == [True, False]
    assert ground.all()
    capture.begin()
    scene.update()
    assert not capture.prohibited.any()
    capture.close()


def test_contact_details_identify_toe_bar_and_exclude_allowed_ground():
    from embodied_learning.tasks.cat_substep_contacts import contact_pairs

    names = ("terrain", "robot/left_foot1_collision", "obstacle_2_1/collision")
    model = SimpleNamespace(geom=lambda i: SimpleNamespace(name=names[i]))
    contact = SimpleNamespace(
        geom=torch.tensor([[1, 0], [1, 2], [1, 2]]),
        worldid=torch.tensor([0, 1, 0]),
        dist=torch.tensor([-0.01, -0.002, 0.1]),
        includemargin=torch.zeros(3),
    )
    data = SimpleNamespace(nacon=torch.tensor([3]), contact=contact)
    assert contact_pairs(model, data) == {
        1: [("robot/left_foot1_collision", "obstacle_2_1/collision")]
    }
