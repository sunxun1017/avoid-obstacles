"""Select paired diagnostic goals from existing cached field indices."""


def route_goal_ids(goals, *, far):
    lanes = {}
    for index, goal in enumerate(goals):
        lanes.setdefault(goal[1:], []).append(index)
    if any(len({goals[i][0] for i in lane}) < 2 for lane in lanes.values()):
        raise ValueError("Each goal lane needs two distinct forward distances")
    choose = max if far else min
    return [choose(lanes[goal[1:]], key=lambda i: goals[i][0]) for goal in goals]


def first_stop_after_landing(moving, landing_time, hz):
    """Time of the first stopped command at or after physical touchdown."""
    return next(
        (
            i / hz
            for i, move in enumerate(moving, start=1)
            if i / hz >= landing_time and not move
        ),
        None,
    )
