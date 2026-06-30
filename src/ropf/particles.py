"""Inertial-particle dynamics in a prescribed carrier flow.

Each particle obeys the linear (Stokes-drag) equation of motion

    dx_p/dt = v_p
    dv_p/dt = (u(x_p, t) - v_p) / tau_p

where ``u`` is the carrier-flow velocity sampled at the particle position
and ``tau_p`` is the particle response time. With tau_f = 1 the Stokes
number is simply St = tau_p / tau_f = tau_p.

All operations are fully vectorised over particles: state arrays have shape
``(N, 2)`` and there are no Python-level loops over individual particles.
"""

from __future__ import annotations

import numpy as np


def particle_rhs(pos, vel, t, tau_p, flow):
    """Right-hand side of the particle ODE system.

    ``flow`` is a callable ``flow(pos, t) -> (..., 2)`` giving the carrier-flow
    velocity. Returns ``(dpos, dvel)`` each of shape ``(N, 2)``.
    """
    u = flow(pos, t)
    dpos = vel
    dvel = (u - vel) / tau_p
    return dpos, dvel


def euler_step(pos, vel, t, dt, tau_p, flow):
    """Explicit forward-Euler step (first order). Kept for comparison."""
    dpos, dvel = particle_rhs(pos, vel, t, tau_p, flow)
    return pos + dt * dpos, vel + dt * dvel


def rk4_step(pos, vel, t, dt, tau_p, flow):
    """Classical fourth-order Runge-Kutta step.

    The carrier flow is time dependent through A(t), so the stages are
    evaluated at t, t + dt/2 and t + dt.
    """
    k1p, k1v = particle_rhs(pos, vel, t, tau_p, flow)
    k2p, k2v = particle_rhs(
        pos + 0.5 * dt * k1p, vel + 0.5 * dt * k1v, t + 0.5 * dt, tau_p, flow
    )
    k3p, k3v = particle_rhs(
        pos + 0.5 * dt * k2p, vel + 0.5 * dt * k2v, t + 0.5 * dt, tau_p, flow
    )
    k4p, k4v = particle_rhs(
        pos + dt * k3p, vel + dt * k3v, t + dt, tau_p, flow
    )
    pos_new = pos + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    vel_new = vel + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return pos_new, vel_new


STEPPERS = {"rk4": rk4_step, "euler": euler_step}


def wrap_periodic(pos, L):
    """Map positions back into the primary domain [0, L) by periodic wrapping.

    Velocities are physical and are never wrapped. Because the carrier flow
    is 2*pi-periodic, wrapping the position has no effect on the sampled
    velocity; it only keeps coordinates bounded for binning and numerics.
    """
    return np.mod(pos, L)
