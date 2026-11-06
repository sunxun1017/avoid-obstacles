"""Parallel active-set Eikonal relaxation on a Cartesian CUDA grid.

This is a numerical backend, not a heap-FMM port. Subcell interface seeds match
the source convention; second-order upwind stencils reduce grid bias. A bounded
solve must converge before its result can be published. Blocked/disconnected
cells remain infinite. CUDA graphs amortize the iterative launch overhead.
"""

import math

import torch
import warp as wp

wp.set_module_options({"enable_backward": False})
BIG = wp.constant(1.0e8)


@wp.func
def neighbor(i: int, axis: int, offset: int, nx: int, ny: int, nz: int):
    strides = wp.vec3i(ny * nz, nz, 1)
    coordinate = wp.vec3i(i // (ny * nz), (i // nz) % ny, i % nz)
    size = wp.vec3i(nx, ny, nz)
    c = coordinate[axis] + offset
    if c < 0 or c >= size[axis]:
        return -1
    return i + offset * strides[axis]


@wp.kernel
def initialize(
    phi: wp.array(dtype=float),
    blocked: wp.array(dtype=wp.bool),
    nx: int,
    ny: int,
    nz: int,
    distance: wp.array(dtype=float),
    fixed: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    fixed[i] = 0
    active[i] = 0
    distance[i] = BIG
    if blocked[i]:
        return
    if phi[i] == 0.0:
        distance[i] = 0.0
        fixed[i] = 1
        return
    inverse = 0.0
    for axis in range(3):
        minimum = float(BIG)
        for side in range(2):
            j = neighbor(i, axis, 2 * side - 1, nx, ny, nz)
            if j >= 0 and not blocked[j] and phi[i] * phi[j] <= 0.0:
                minimum = wp.min(minimum, wp.abs(phi[i] / (phi[i] - phi[j])))
        if minimum < BIG:
            inverse += 1.0 / (minimum * minimum)
    if inverse > 0.0:
        distance[i] = 1.0 / wp.sqrt(inverse)
        fixed[i] = 1
    else:
        active[i] = 1


@wp.func
def value_at(
    j: int,
    i: int,
    phi: wp.array(dtype=float),
    blocked: wp.array(dtype=wp.bool),
    distance: wp.array(dtype=float),
):
    if j < 0:
        return float(BIG)
    if blocked[j] or phi[i] * phi[j] < 0.0:
        return float(BIG)
    return distance[j]


@wp.kernel
def relax(
    phi: wp.array(dtype=float),
    blocked: wp.array(dtype=wp.bool),
    fixed: wp.array(dtype=wp.int32),
    old: wp.array(dtype=float),
    new: wp.array(dtype=float),
    active: wp.array(dtype=wp.int32),
    next_active: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    tolerance: float,
    order: int,
):
    i = wp.tid()
    previous = old[i]
    new[i] = previous
    if blocked[i] or fixed[i] != 0 or active[i] == 0:
        return
    a = wp.vec3(BIG)
    weights = wp.vec3(1.0)
    for axis in range(3):
        left = value_at(neighbor(i, axis, -1, nx, ny, nz), i, phi, blocked, old)
        right = value_at(neighbor(i, axis, 1, nx, ny, nz), i, phi, blocked, old)
        sign = -1
        first = left
        if right < left:
            sign = 1
            first = right
        a[axis] = first
        if order == 2 and first < BIG:
            second = value_at(
                neighbor(i, axis, 2 * sign, nx, ny, nz), i, phi, blocked, old
            )
            if second <= first:
                a[axis] = (4.0 * first - second) / 3.0
                weights[axis] = 2.25
    # Sort axis contributions; only causal axes enter the quadratic.
    for k in range(2):
        for j in range(2 - k):
            if a[j] > a[j + 1]:
                temporary = a[j]
                a[j] = a[j + 1]
                a[j + 1] = temporary
                temporary = weights[j]
                weights[j] = weights[j + 1]
                weights[j + 1] = temporary
    if a[0] >= BIG:
        return
    solution = a[0] + 1.0 / wp.sqrt(weights[0])
    # Shift by the smallest neighbor to avoid catastrophic cancellation for
    # distances hundreds of cells from the seed interface.
    base = a[0]
    A = weights[0]
    B = 0.0
    C = -1.0
    for axis in range(1, 3):
        if solution > a[axis]:
            delta = a[axis] - base
            A += weights[axis]
            B += weights[axis] * delta
            C += weights[axis] * delta * delta
            solution = base + (B + wp.sqrt(wp.max(0.0, B * B - A * C))) / A
    candidate = wp.min(previous, solution)
    new[i] = candidate
    if previous - candidate > tolerance:
        wp.atomic_or(next_active, i, 1)
        for axis in range(3):
            for offset in range(-2, 3):
                j = neighbor(i, axis, offset, nx, ny, nz)
                if j >= 0:
                    wp.atomic_or(next_active, j, 1)


class CudaEikonal:
    """Reusable workspace; solves are isolated, including obstacle removals."""

    def __init__(self, shape, device):
        if len(shape) != 3 or min(shape) < 3:
            raise ValueError("Expected a three-dimensional grid with axes >=3.")
        self.shape = tuple(shape)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA Eikonal requires a CUDA device.")
        wp.init()
        self.size = math.prod(shape)
        self.phi = torch.empty(self.shape, device=device)
        self.blocked = torch.empty(self.shape, dtype=torch.bool, device=device)
        self.fixed = torch.empty(self.size, dtype=torch.int32, device=device)
        self.distances = [torch.empty(self.size, device=device) for _ in range(2)]
        self.active = [
            torch.empty(self.size, dtype=torch.int32, device=device)
            for _ in range(2)
        ]
        self._graphs = {}
        self.iterations = 0
        self.stream = torch.cuda.Stream(device=self.device)

    def _step(self, old, new, *, tolerance, order):
        wp.from_torch(self.active[new]).zero_()
        wp.launch(
            relax,
            dim=self.size,
            inputs=[
                wp.from_torch(self.phi.flatten()),
                wp.from_torch(self.blocked.flatten()),
                wp.from_torch(self.fixed),
                wp.from_torch(self.distances[old]),
                wp.from_torch(self.distances[new]),
                wp.from_torch(self.active[old]),
                wp.from_torch(self.active[new]),
                *self.shape,
                tolerance,
                order,
            ],
        )

    @torch.no_grad()
    def solve(
        self,
        phi,
        blocked,
        *,
        spacing,
        max_iterations=None,
        tolerance=1.0e-4,
        order=2,
    ):
        caller = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            result = self._solve(
                phi,
                blocked,
                spacing=spacing,
                max_iterations=max_iterations,
                tolerance=tolerance,
                order=order,
            )
        caller.wait_stream(self.stream)
        result.record_stream(caller)
        return result

    def _solve(self, phi, blocked, *, spacing, max_iterations, tolerance, order):
        if (
            phi.shape != self.shape
            or blocked.shape != self.shape
            or phi.device != self.phi.device
            or blocked.device != phi.device
            or phi.dtype != torch.float32
            or blocked.dtype != torch.bool
            or not math.isfinite(spacing)
            or spacing <= 0
            or not math.isfinite(tolerance)
            or tolerance <= 0
            or order not in (1, 2)
            or not torch.isfinite(phi).all()
        ):
            raise ValueError(
                "Invalid Eikonal grid, precision, device or parameters."
            )
        limit = (
            max_iterations
            if max_iterations is not None
            else 2 * sum(self.shape) + 64
        )
        if type(limit) is not int or limit < 1:
            raise ValueError("max_iterations must be a positive integer.")
        self.phi.copy_(phi)
        self.blocked.copy_(blocked)
        with wp.ScopedStream(
            wp.stream_from_torch(torch.cuda.current_stream(phi.device))
        ):
            wp.launch(
                initialize,
                dim=self.size,
                inputs=[
                    wp.from_torch(self.phi.flatten()),
                    wp.from_torch(self.blocked.flatten()),
                    *self.shape,
                    wp.from_torch(self.distances[0]),
                    wp.from_torch(self.fixed),
                    wp.from_torch(self.active[0]),
                ],
            )
            if not self.fixed.any():
                raise ValueError("Eikonal field has no zero interface.")
            key = (tolerance, order)
            if key not in self._graphs:
                # Preload compilation before graph capture.
                wp.load_module(module=__name__, device=str(phi.device))
                with wp.ScopedCapture() as capture:
                    for j in range(32):
                        self._step(
                            j % 2, (j + 1) % 2, tolerance=tolerance, order=order
                        )
                self._graphs[key] = capture.graph
            count = 0
            current = 0
            while count < limit:
                if limit - count >= 32 and current == 0:
                    wp.capture_launch(self._graphs[key])
                    count += 32
                else:
                    self._step(
                        current, 1 - current, tolerance=tolerance, order=order
                    )
                    current = 1 - current
                    count += 1
                if not self.active[current].any():
                    break
            else:
                raise ValueError(
                    f"Eikonal solve did not converge in {limit} iterations."
                )
        self.iterations = count
        distance = self.distances[current].reshape(self.shape)
        return torch.where(
            distance < 1.0e7, distance * torch.sign(phi) * spacing, torch.inf
        )
