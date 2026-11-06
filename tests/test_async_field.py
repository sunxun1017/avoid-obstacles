"""Process-backed fields: timing and generation correctness without simulation."""

import time

import pytest
import torch

from embodied_learning.perception.async_field import AsyncFieldMap
from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.rays import RayMeasurements


def frame(batch=2):
    origin = torch.tensor([[[0.213, 0.217, 0.413]]]).repeat(batch, 1, 1)
    end = origin.clone()
    end[..., 0] = 0.69
    mask = torch.ones(batch, 1, dtype=torch.bool)
    return RayMeasurements(origin, end, mask, mask)


def goals(batch=2):
    return torch.tensor([[0.413, 0.427, 0.613]]).repeat(batch, 1)


def make_pipeline():
    return AsyncFieldMap(
        OccupancyMapper(2, (20, 20, 25), voxel_size=0.04),
        max_rays=4,
        worker_device="cpu",
        preprocess=False,
        update_period=0.01,
        max_age=0.5,
    )


def wait_result(p, *, timeout=5.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if p.poll():
            return
        time.sleep(0.005)
    pytest.fail(f"Worker did not publish: {p.diagnostics()}")


def test_worker_publishes_field_with_original_measurement_time_and_cleans_up():
    p = make_pipeline()
    try:
        p.start()
        stamp = time.monotonic()
        assert p.update(frame(), goals(), timestamp=stamp)
        wait_result(p)
        query = p.sample(frame().origins, timestamp=stamp + 0.05)
        assert query.sample.guidance_valid.all()
        torch.testing.assert_close(
            query.age, torch.full((2,), 0.05, dtype=torch.float64)
        )
        assert not p.sample(
            frame().origins, timestamp=stamp + 0.6
        ).sample.guidance_valid.any()
        assert p.diagnostics()["build_counts"] == [1, 1]
    finally:
        p.close()
        p.close()
    assert not p.worker_alive


def test_reset_and_goal_change_reject_results_already_in_flight():
    with make_pipeline() as p:
        stamp = time.monotonic()
        p.update(frame(), goals(), timestamp=stamp)
        p.reset(torch.tensor([0]))
        wait_result(p)
        assert p.versions.tolist() == [0, 1]
        target = goals()
        target[1, 0] = 10.0
        p.update(frame(), target, timestamp=time.monotonic())
        # Changing a goal invalidates that field before a worker result exists.
        assert (
            not p.sample(frame().origins, timestamp=time.monotonic())
            .sample.guidance_valid[1]
            .any()
        )
        wait_result(p)
        assert p.versions[0] == 1
        assert "goal" in p.errors[1]
        assert (
            not p.sample(frame().origins, timestamp=time.monotonic())
            .sample.guidance_valid[1]
            .any()
        )


def test_consumer_does_not_wait_on_busy_mailboxes_and_keeps_latest_scan():
    with make_pipeline() as p:
        p._transport.input_lock.acquire()
        try:
            begin = time.perf_counter()
            assert not p.update(frame(), goals(), timestamp=time.monotonic())
            assert time.perf_counter() - begin < 0.05
        finally:
            p._transport.input_lock.release()
        p._transport.output_lock.acquire()
        try:
            begin = time.perf_counter()
            assert not p.poll()
            assert time.perf_counter() - begin < 0.05
            p.update(frame(), goals(), timestamp=time.monotonic())
            # Worker may compute one request, but cannot publish it yet.
            time.sleep(0.1)
            target = goals()
            target[:, 0] += 0.04
            p.update(frame(), target, timestamp=time.monotonic())
            target[:, 0] += 0.04
            p.update(frame(), target, timestamp=time.monotonic())
        finally:
            p._transport.output_lock.release()
        wait_result(p)
        assert p.sample(
            frame().origins, timestamp=time.monotonic()
        ).sample.guidance_valid.all()
        assert (
            p.diagnostics()["published_sequence"]
            == p.diagnostics()["submitted_sequence"]
        )


def test_worker_failure_invalidates_fields_and_reports_error():
    with make_pipeline() as p:
        p.update(frame(), goals(), timestamp=time.monotonic())
        wait_result(p)
        p._process.terminate()
        p._process.join(2.0)
        with pytest.raises(RuntimeError, match="worker"):
            p.poll()
        assert not p.sample(
            frame().origins, timestamp=time.monotonic()
        ).sample.guidance_valid.any()


def test_future_measurement_and_future_publication_are_rejected():
    with make_pipeline() as p:
        with pytest.raises(ValueError, match="future"):
            p.update(frame(), goals(), timestamp=time.monotonic() + 60.0)
        p.update(frame(), goals(), timestamp=time.monotonic())
        deadline = time.monotonic() + 5.0
        while int(p._transport.arrays["out_sequence"][0]) == 0:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        with p._transport.output_lock:
            p._transport.arrays["out_stamp"][:] = time.monotonic() + 60.0
        assert not p.poll()
        assert p.versions.tolist() == [0, 0]


def test_overwritten_scan_does_not_starve_phase_offset_environment():
    from types import SimpleNamespace

    p = AsyncFieldMap(
        OccupancyMapper(2, (20, 20, 25), voxel_size=0.04),
        max_rays=4,
        preprocess=False,
        update_period=0.1,
        max_age=0.5,
    )
    with p:
        base = time.monotonic() - 10.0
        target = goals()
        p.update(frame(), target, timestamp=base)
        wait_result(p)
        # Hold notifications, so A's pending goal-change scan is overwritten by
        # B's scan before the worker reads either. Their schedules now differ.
        wake = p._transport.sync["wake"]
        p._transport.sync["wake"] = SimpleNamespace(release=lambda: None)
        try:
            target[0, 0] += 0.04
            assert p.update(frame(), target, timestamp=base + 0.05)
            assert p.update(frame(), target, timestamp=base + 0.1)
        finally:
            p._transport.sync["wake"] = wake
            wake.release()
        wait_result(p)
        for step in range(3, 11):
            assert p.update(frame(), target, timestamp=base + step * 0.05)
            wait_result(p)
        assert p.versions.tolist() == [5, 6]
        assert p.sample(
            frame().origins, timestamp=base + 0.5
        ).sample.guidance_valid.all()


def test_transport_unlinks_shared_memory_when_primitive_creation_fails(monkeypatch):
    from types import SimpleNamespace

    from embodied_learning.perception import field_worker

    real = field_worker.shared_memory.SharedMemory
    names = []

    def allocate(*args, **kwargs):
        memory = real(*args, **kwargs)
        names.append(memory.name)
        return memory

    def fail_lock():
        raise OSError("injected semaphore exhaustion")

    monkeypatch.setattr(field_worker.shared_memory, "SharedMemory", allocate)
    with pytest.raises(OSError, match="semaphore"):
        field_worker.FieldTransport(
            field_worker.layout(1, (3, 3, 3), 1),
            context=SimpleNamespace(Lock=fail_lock),
        )
    assert len(names) == 1
    try:
        leaked = real(name=names[0])
    except FileNotFoundError:
        exists = False
    else:
        exists = True
        leaked.close()
        leaked.unlink()
    assert not exists, "Constructor failure leaked its shared-memory allocation."


def test_start_cleans_transport_when_process_construction_fails(monkeypatch):
    from multiprocessing import shared_memory

    p = make_pipeline()

    def fail_process(**kwargs):
        raise OSError("injected process construction failure")

    monkeypatch.setattr(p._context, "Process", fail_process)
    try:
        with pytest.raises(OSError, match="process construction"):
            p.start()
        assert p._closed
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=p._transport.memory.name)
    finally:
        p.close()
