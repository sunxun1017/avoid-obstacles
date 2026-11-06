"""Select paired diagnostic goals from existing cached field indices."""


def route_goal_ids(goals, *, far):
    lanes = {}
    for index, goal in enumerate(goals):
        lanes.setdefault(goal[1:], []).append(index)
    if any(len({goals[i][0] for i in lane}) < 2 for lane in lanes.values()):
        raise ValueError("Each goal lane needs two distinct forward distances")
    choose = max if far else min
    return [choose(lanes[goal[1:]], key=lambda i: goals[i][0]) for goal in goals]
