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


def gate_from_evaluations(reports):
    """Join two held-out cohorts with one fixed checkpoint and stage."""
    if len(reports) != 2:
        raise ValueError("Expected exactly two evaluation reports")
    first = reports[0]
    stage = first["training_stage"]
    digest = first["checkpoint_sha256"]
    provenance = ("protocol_sha256", "evaluation_code_commit", "training_seed")
    cohorts = {}
    for report in reports:
        result = report["terminal_v1"]
        if (
            report["training_stage"] != stage
            or report["evaluation_stage"] != stage
            or report["checkpoint_sha256"] != digest
            or report["completed_updates"] != 600
            or report["execution_profile"] != "source_train"
            or report["contact_sampling"] != "physics_substep"
            or result["protocol"] != VERSION
            or "diagnostic_override" in report
            or report["seed"] in cohorts
            or any(report[key] != first[key] for key in provenance)
        ):
            raise ValueError("Reports must use the same update-600 checkpoint, stage and full protocol")
        cohorts[report["seed"]] = (result["success_count"], result["episodes"])
    return dict(
        protocol=VERSION, stage=stage, checkpoint_sha256=digest,
        **{key: first[key] for key in provenance},
        completed_updates=600, passed=promotion_passed(600, cohorts),
        success_count=sum(s for s, _ in cohorts.values()), episodes=sum(n for _, n in cohorts.values()),
        by_seed={seed: {"success_count": s, "episodes": n} for seed, (s, n) in cohorts.items()},
    )
