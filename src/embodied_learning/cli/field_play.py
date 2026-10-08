"""Inference-only native/loopback Viser playback; owns environment cleanup."""

from embodied_learning.protocols.imitation import (
    ExpertPolicy,
    FieldTrainingEnvironment,
)


def play_policy(
    env: FieldTrainingEnvironment,
    policy: ExpertPolicy,
    *,
    viewer: str,
    port: int,
    steps: int | None = None,
) -> None:
    """Run deterministic predictions and close resources even on setup failure."""
    server = None
    display = None
    try:
        if viewer not in ("native", "viser") or not 1 <= port <= 65535:
            raise ValueError("Invalid playback viewer/port.")
        if steps is not None and steps < 1:
            raise ValueError("Playback steps must be positive.")
        from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

        if viewer == "viser":
            import viser

            server = viser.ViserServer(host="127.0.0.1", port=port)
            display = ViserPlayViewer(env, policy.predict, viser_server=server)
        else:
            display = NativeMujocoViewer(env, policy.predict)
        # BaseViewer.run closes internally; make cleanup idempotent for setup
        # failures and avoid removing GUI handles twice after a normal run.
        close = display.close
        closed = False

        def close_once():
            nonlocal closed
            if not closed:
                closed = True
                close()

        display.close = close_once
        display.run(num_steps=steps)
    finally:
        try:
            if display is not None:
                display.close()
        finally:
            try:
                if server is not None:
                    server.stop()
            finally:
                env.close()
