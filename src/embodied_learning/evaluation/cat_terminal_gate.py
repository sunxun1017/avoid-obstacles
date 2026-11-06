"""CAT v1 fixed-horizon settlement protocol, independent of training rewards."""

VERSION = "cat_terminal_20s_v1"
HORIZON_STEPS = 1000
HOLD_STEPS = 25


def update_hold(ordinary, strict, distance, linear_speed, angular_speed):
    """Count consecutive qualifying control transitions (scalars or tensors)."""
    near = distance <= 0.1
    slow = near & (linear_speed < 0.2) & (angular_speed < 0.2)
    very_slow = near & (linear_speed < 0.1) & (angular_speed < 0.1)
    return (ordinary + 1) * slow, (strict + 1) * very_slow


def classify_terminal(steps, ordinary, strict, *, invalid, fall, contact, reached):
    if invalid:
        return False, False, "invalid"
    if fall:
        return False, False, "fall"
    if contact:
        return False, False, "prohibited_contact"
    if steps != HORIZON_STEPS:
        raise ValueError("Clean episode must run to the fixed horizon")
    if ordinary >= HOLD_STEPS:
        return True, strict >= HOLD_STEPS, None
    return False, False, "not_settled" if reached else "not_reached"


def promotion_passed(update, cohorts):
    """Read only the two seed-specific (successes, episodes) v1 gate results."""
    if update != 600 or len(cohorts) != 2:
        raise ValueError("Expected update 600 and two distinct evaluation seeds")
    if any(total != 128 or not 0 <= successes <= total for successes, total in cohorts.values()):
        raise ValueError("Expected 128 evaluated episodes per seed and valid counts")
    return sum(successes for successes, _ in cohorts.values()) >= 226
