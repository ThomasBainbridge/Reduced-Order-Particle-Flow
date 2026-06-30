"""Physics sanity checks for the dataset-generation pipeline.

Runnable either with pytest (``pytest -q``) or directly
(``python tests/test_physics.py``).
"""

import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import dataclasses as _dc  # noqa: E402

from ropf import SimConfig, concentration_field, simulate_case  # noqa: E402
from ropf.carrier_flow import (  # noqa: E402
    RandomFourierFlow,
    TaylorGreenFlow,
    taylor_green_velocity,
    velocity_divergence,
)
from ropf.particles import rk4_step  # noqa: E402


def test_carrier_flow_is_divergence_free():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 2 * np.pi, size=(2000, 2))
    flow = TaylorGreenFlow()
    for t in [0.0, 1.3, 7.7]:
        div = velocity_divergence(flow, pos, t)
        assert np.max(np.abs(div)) < 1e-4


def test_fourier_flow_is_divergence_free_and_periodic():
    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 2 * np.pi, size=(1000, 2))
    flow = RandomFourierFlow(n_modes=8, max_wavenumber=4, seed=1)
    for t in [0.0, 2.1, 5.5]:
        assert np.max(np.abs(velocity_divergence(flow, pos, t))) < 1e-3
    shifted = pos + np.array([2 * np.pi, -2 * np.pi])
    assert np.allclose(flow(pos, 1.0), flow(shifted, 1.0), atol=1e-9)


def test_carrier_flow_periodicity():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 2 * np.pi, size=(500, 2))
    shifted = pos + np.array([2 * np.pi, -2 * np.pi])
    u0 = taylor_green_velocity(pos, 0.7)
    u1 = taylor_green_velocity(shifted, 0.7)
    assert np.allclose(u0, u1, atol=1e-9)


def test_concentration_field_is_normalised():
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 2 * np.pi, size=(5000, 2))
    field = concentration_field(pos, 64, 64, 2 * np.pi)
    assert field.shape == (64, 64)
    assert np.isclose(field.sum(), 1.0, atol=1e-6)
    assert np.all(field >= 0)


def test_uniform_particles_give_flat_field():
    # Particles on a regular grid should bin to a near-uniform field.
    n = 64
    xs = (np.arange(n) + 0.5) * (2 * np.pi / n)
    gx, gy = np.meshgrid(xs, xs)
    pos = np.column_stack([gx.ravel(), gy.ravel()])
    field = concentration_field(pos, n, n, 2 * np.pi)
    assert np.allclose(field, 1.0 / (n * n))


def test_low_stokes_tracks_fluid_better_than_high_stokes():
    """St=0.1 particles should follow the carrier flow more closely than
    St=10 particles. We measure the rms slip |v_p - u| at the end of a short
    run; the low-St case must have substantially smaller slip."""
    cfg = dataclasses.replace(
        SimConfig(), n_particles=2000, t_final=2.0, dt=0.01,
        save_interval=0.1, nx=32, ny=32,
    )

    flow = cfg.make_flow()

    def final_slip(stokes):
        rng = np.random.default_rng(0)
        L = cfg.domain_size
        pos = rng.uniform(0, L, size=(cfg.n_particles, 2))
        vel = flow(pos, 0.0)
        t = 0.0
        for i in range(1, cfg.n_steps + 1):
            pos, vel = rk4_step(pos, vel, t, cfg.dt, stokes, flow)
            t = i * cfg.dt
        u = flow(pos, t)
        return np.sqrt(np.mean((vel - u) ** 2))

    slip_low = final_slip(0.1)
    slip_high = final_slip(10.0)
    assert slip_low < slip_high
    assert slip_low < 0.3 * slip_high


def test_simulate_case_shapes_and_conservation():
    cfg = dataclasses.replace(
        SimConfig(), n_particles=1500, t_final=1.0, dt=0.02,
        save_interval=0.1, nx=32, ny=32,
    )
    fields, times, particles = simulate_case(1.0, 0, cfg, return_particles=True)
    assert fields.shape == (cfg.n_saves, 1, cfg.ny, cfg.nx)
    assert times.shape == (cfg.n_saves,)
    assert particles.shape == (cfg.n_saves, cfg.n_particles, 2)
    # Every snapshot conserves particle mass.
    sums = fields.sum(axis=(1, 2, 3))
    assert np.allclose(sums, 1.0, atol=1e-5)
    # Positions stay inside the periodic domain.
    assert particles.min() >= 0.0
    assert particles.max() <= cfg.domain_size + 1e-5


def test_smoothing_preserves_mass_and_reduces_noise():
    rng = np.random.default_rng(7)
    pos = rng.uniform(0, 2 * np.pi, size=(5000, 2))
    raw = concentration_field(pos, 64, 64, 2 * np.pi, smoothing=0.0)
    smooth = concentration_field(pos, 64, 64, 2 * np.pi, smoothing=1.5)
    assert np.isclose(smooth.sum(), 1.0, atol=1e-5)        # mass conserved
    assert smooth.var() < raw.var()                        # noise suppressed


def test_fourier_flow_simulation_runs():
    cfg = _dc.replace(
        SimConfig(), flow_type="fourier", n_fourier_modes=6, n_particles=1000,
        t_final=1.0, dt=0.02, save_interval=0.1, nx=32, ny=32,
    )
    fields, times, _ = simulate_case(1.0, 0, cfg)
    assert fields.shape == (cfg.n_saves, 1, cfg.ny, cfg.nx)
    assert np.allclose(fields.sum(axis=(1, 2, 3)), 1.0, atol=1e-5)


def test_reproducibility():
    cfg = dataclasses.replace(
        SimConfig(), n_particles=1000, t_final=0.5, dt=0.05,
        save_interval=0.1, nx=16, ny=16,
    )
    a, _, _ = simulate_case(1.0, 42, cfg)
    b, _, _ = simulate_case(1.0, 42, cfg)
    assert np.array_equal(a, b)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} physics tests passed.")
