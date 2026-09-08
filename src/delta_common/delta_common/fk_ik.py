"""Delta robot forward kinematics (pure Python / numpy, no ROS deps).

Geometry convention (matches nodennbot_cad_description/urdf/nodennbot.urdf.xacro):
  * 3 arms at azimuths 0, 120, 240 deg.
  * Shoulder i sits at radius `base_radius` along its azimuth, on the base plane z=0.
  * Shoulder axis is tangential (local Y). Angle theta rotates the bicep about it;
    theta=0 => bicep horizontal, pointing radially outward. Positive theta lowers
    the elbow (rotation about +Y sends +X toward -Z).
  * Forearm is treated as a single rod of length `forearm_length` from elbow to the
    platform attach point (radius `platform_radius` from the platform centre).

FK reduces to intersecting 3 spheres of radius `forearm_length` centred at the
elbow points shifted inward by `platform_radius` (classic delta trilateration).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeltaGeometry:
    base_radius: float = 0.16321    # R: base centre -> shoulder axis (from CAD)
    # The CAD implies r ~= 0.0303 (from its home platform drop of 0.22175 and its
    # forearm pitch of 0.5876 rad, both with l=0.400). Kept at 0.035 pending a
    # physical measurement of the platform centre -> forearm ball distance.
    platform_radius: float = 0.035  # r: platform centre -> forearm attach
    bicep_length: float = 0.200     # L: shoulder -> elbow
    forearm_length: float = 0.400   # l: elbow -> platform
    # Rotated 180 deg from (0,120,240) to match the physical robot's mounting
    # (real x,y were opposite to RViz); confirmed by the CAD export. MUST stay in
    # sync with the arm angles in
    # nodennbot_cad_description/urdf/nodennbot.urdf.xacro.
    azimuths_deg: tuple = (180.0, 300.0, 60.0)

    @property
    def azimuths(self) -> list:
        return [math.radians(a) for a in self.azimuths_deg]


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def elbow_position(geom: DeltaGeometry, i: int, theta: float) -> np.ndarray:
    """World position of arm i's elbow given shoulder angle theta."""
    phi = geom.azimuths[i]
    radial = geom.base_radius + geom.bicep_length * math.cos(theta)
    z = -geom.bicep_length * math.sin(theta)
    return np.array([radial * math.cos(phi), radial * math.sin(phi), z])


def _virtual_centre(geom: DeltaGeometry, i: int, theta: float) -> np.ndarray:
    """Elbow shifted inward by platform_radius: platform centre is `forearm_length`
    away from this point (that inward shift folds the platform offset into the sphere)."""
    phi = geom.azimuths[i]
    d = np.array([math.cos(phi), math.sin(phi), 0.0])
    return elbow_position(geom, i, theta) - geom.platform_radius * d


def forward_kinematics(geom: DeltaGeometry, thetas) -> np.ndarray | None:
    """Platform centre position for shoulder angles (t1, t2, t3).

    Returns the lower of the two trilateration roots, or None if the configuration
    is unreachable (spheres don't intersect) or degenerate.
    """
    c1 = _virtual_centre(geom, 0, thetas[0])
    c2 = _virtual_centre(geom, 1, thetas[1])
    c3 = _virtual_centre(geom, 2, thetas[2])
    l = geom.forearm_length

    # Two planes from pairwise sphere differences: 2(cj-c1).p = |cj|^2 - |c1|^2
    A = 2.0 * (c2 - c1)
    B = 2.0 * (c3 - c1)
    bA = c2 @ c2 - c1 @ c1
    bB = c3 @ c3 - c1 @ c1

    d = np.cross(A, B)  # direction of the intersection line
    nd2 = float(d @ d)
    if nd2 < 1e-12:
        return None  # arms collinear / degenerate

    # Particular point on the line: solve [A; B; d] p0 = [bA; bB; 0].
    M = np.vstack([A, B, d])
    try:
        p0 = np.linalg.solve(M, np.array([bA, bB, 0.0]))
    except np.linalg.LinAlgError:
        return None

    # p = p0 + t d, substitute into |p - c1|^2 = l^2 -> quadratic in t.
    w = p0 - c1
    qa = nd2
    qb = 2.0 * float(d @ w)
    qc = float(w @ w) - l * l
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return None  # out of workspace
    sq = math.sqrt(disc)
    pa = p0 + ((-qb + sq) / (2.0 * qa)) * d
    pb = p0 + ((-qb - sq) / (2.0 * qa)) * d
    return pa if pa[2] < pb[2] else pb  # physical solution hangs below the base


def passive_angles(geom: DeltaGeometry, i: int, theta: float, platform: np.ndarray):
    """(elbow_yaw, elbow_pitch) that point arm i's forearm from its elbow to the
    platform attach point, expressed in the bicep frame.

    Matches the URDF universal joint order: yaw about Z, then pitch about Y, so
    forearm_x = Rz(yaw) Ry(pitch) . x_hat.
    """
    phi = geom.azimuths[i]
    attach = platform + geom.platform_radius * np.array([math.cos(phi), math.sin(phi), 0.0])
    u = attach - elbow_position(geom, i, theta)
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return 0.0, 0.0
    u = u / n
    # Express the world direction in the bicep frame: R_wb = Rz(phi) Ry(theta).
    ub = _rot_y(-theta) @ (_rot_z(-phi) @ u)
    pitch = -math.asin(max(-1.0, min(1.0, ub[2])))
    yaw = math.atan2(ub[1], ub[0])
    return yaw, pitch


def inverse_kinematics(geom: DeltaGeometry, platform):
    """Shoulder angles (t1, t2, t3) that place the platform centre at `platform`.

    Each arm is solved independently: the elbow lies on the bicep circle (radius
    bicep_length about the shoulder, in the arm's radial/vertical plane) and must be
    `forearm_length` from the attach point. The attach point's tangential (out-of-
    plane) offset shrinks the in-plane forearm reach, then it's a circle-circle
    intersection. Returns None if any arm can't reach.
    """
    P = np.asarray(platform, dtype=float)
    L = geom.bicep_length
    thetas = []
    for i in range(3):
        phi = geom.azimuths[i]
        c, s = math.cos(phi), math.sin(phi)
        # attach point in the arm's mount frame: (radial, tangential, z)
        a_rho = P[0] * c + P[1] * s + geom.platform_radius
        a_tau = -P[0] * s + P[1] * c
        a_z = P[2]

        # tangential offset uses up part of the forearm; remainder spans the plane
        lp2 = geom.forearm_length ** 2 - a_tau ** 2
        if lp2 < 0.0:
            return None
        lp = math.sqrt(lp2)

        # circle-circle: shoulder circle S=(R,0) r=L ; forearm circle B=(a_rho,a_z) r=lp
        dx = a_rho - geom.base_radius
        dv = a_z
        d = math.hypot(dx, dv)
        if d < 1e-9 or d > L + lp or d < abs(L - lp):
            return None
        a = (L * L - lp * lp + d * d) / (2.0 * d)
        h2 = L * L - a * a
        if h2 < 0.0:
            return None
        h = math.sqrt(h2)
        ux, uv = dx / d, dv / d           # unit shoulder->B
        mx, mv = geom.base_radius + a * ux, a * uv
        px, pv = -uv, ux                  # perpendicular
        e1 = (mx + h * px, mv + h * pv)
        e2 = (mx - h * px, mv - h * pv)
        E = e1 if e1[0] >= e2[0] else e2  # elbow "knee-out": larger radial coord
        thetas.append(math.atan2(-E[1], E[0] - geom.base_radius))
    return thetas


# The 3 actuated joints (the real DOF), in arm order. Used by the cmd/state contract.
SHOULDER_NAMES = ["arm1_shoulder", "arm2_shoulder", "arm3_shoulder"]

# Joint names, in the order this module reports them. Must match the URDF.
JOINT_NAMES = [
    "arm1_shoulder", "arm1_elbow_yaw", "arm1_elbow_pitch",
    "arm2_shoulder", "arm2_elbow_yaw", "arm2_elbow_pitch",
    "arm3_shoulder", "arm3_elbow_yaw", "arm3_elbow_pitch",
    "platform_x", "platform_y", "platform_z",
]


def joint_state(geom: DeltaGeometry, thetas):
    """Full name->value dict for all 12 joints, or None if unreachable."""
    platform = forward_kinematics(geom, thetas)
    if platform is None:
        return None
    values = {}
    for i in range(3):
        yaw, pitch = passive_angles(geom, i, thetas[i], platform)
        values[f"arm{i + 1}_shoulder"] = float(thetas[i])
        values[f"arm{i + 1}_elbow_yaw"] = float(yaw)
        values[f"arm{i + 1}_elbow_pitch"] = float(pitch)
    values["platform_x"] = float(platform[0])
    values["platform_y"] = float(platform[1])
    values["platform_z"] = float(platform[2])
    return values


# ─────────────────────────────────────────────────────────────────────────
# mm / degree adapter
#
# forward_kinematics()/inverse_kinematics() above work natively in metres and
# radians. Every other package in this repo (motor CAN feedback in degrees,
# camera calibration, workspace/pick-place config) works in millimetres, so
# this thin boundary layer converts once, here, instead of scattering
# mm<->m conversions across every caller.
# ─────────────────────────────────────────────────────────────────────────
_GEOM = DeltaGeometry()


def solve_fk_mm(theta1_deg: float, theta2_deg: float, theta3_deg: float):
    """Platform position for shoulder angles given in degrees.

    Returns (ok, x_mm, y_mm, z_mm). ok=False if the configuration is
    unreachable/degenerate.
    """
    thetas = [math.radians(theta1_deg), math.radians(theta2_deg), math.radians(theta3_deg)]
    p = forward_kinematics(_GEOM, thetas)
    if p is None:
        return False, 0.0, 0.0, 0.0
    return True, float(p[0]) * 1000.0, float(p[1]) * 1000.0, float(p[2]) * 1000.0


def solve_ik_mm(x_mm: float, y_mm: float, z_mm: float):
    """Shoulder angles (degrees) that place the platform at (x_mm, y_mm, z_mm).

    Returns (ok, theta1_deg, theta2_deg, theta3_deg). ok=False if the target
    is unreachable.
    """
    platform_m = (x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0)
    thetas = inverse_kinematics(_GEOM, platform_m)
    if thetas is None:
        return False, 0.0, 0.0, 0.0
    return True, math.degrees(thetas[0]), math.degrees(thetas[1]), math.degrees(thetas[2])


def check_workspace(x_mm: float, y_mm: float, z_mm: float) -> bool:
    """Return True if (x, y, z) in mm is within the configured robot workspace.

    The bounding box (X_LIMIT / Y_LIMIT / Z_MIN / Z_MAX) is a rectangular
    over-approximation.  Corners of the box (large XY + Z near ceiling) are
    outside the actual reachable sphere, so we also verify IK feasibility.
    """
    from delta_common import config

    if not (
        abs(x_mm) <= config.X_LIMIT
        and abs(y_mm) <= config.Y_LIMIT
        and config.Z_MIN <= z_mm <= config.Z_MAX
    ):
        return False
    ok, *_ = solve_ik_mm(x_mm, y_mm, z_mm)
    return ok


if __name__ == "__main__":
    # Self-test: verify FK + passive angles are self-consistent.
    g = DeltaGeometry()

    # 1) Home pose (all shoulders horizontal) -> platform straight below.
    P = forward_kinematics(g, [0.0, 0.0, 0.0])
    assert P is not None, "home should be reachable"
    print(f"home platform = {P}")
    assert abs(P[0]) < 1e-9 and abs(P[1]) < 1e-9, "home platform must be centred"
    expected_z = -math.sqrt(g.forearm_length**2 - (g.base_radius + g.bicep_length - g.platform_radius)**2)
    assert abs(P[2] - expected_z) < 1e-9, f"z {P[2]} != {expected_z}"

    # 2) For several configs, reconstruct the forearm tip from the passive angles
    #    and confirm it lands on the platform attach point (closed-loop consistency).
    def forearm_tip(geom, i, theta, yaw, pitch):
        phi = geom.azimuths[i]
        Rwb = _rot_z(phi) @ _rot_y(theta)                 # bicep frame in world
        fdir_bicep = _rot_z(yaw) @ _rot_y(pitch) @ np.array([1.0, 0.0, 0.0])
        return elbow_position(geom, i, theta) + Rwb @ (geom.forearm_length * fdir_bicep)

    for thetas in ([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, -0.1, 0.4], [-0.2, 0.5, 0.1]):
        P = forward_kinematics(g, thetas)
        assert P is not None, f"{thetas} unreachable"
        for i in range(3):
            yaw, pitch = passive_angles(g, i, thetas[i], P)
            phi = g.azimuths[i]
            attach = P + g.platform_radius * np.array([math.cos(phi), math.sin(phi), 0.0])
            tip = forearm_tip(g, i, thetas[i], yaw, pitch)
            err = float(np.linalg.norm(tip - attach))
            assert err < 1e-9, f"thetas={thetas} arm{i}: tip {tip} != attach {attach} (err {err})"
        print(f"thetas={thetas} -> platform {np.round(P, 4)}  OK")

    # 3) Round trip: for a grid of reachable targets, IK then FK should recover P.
    tested = 0
    for x in (-0.06, -0.03, 0.0, 0.03, 0.06):
        for y in (-0.06, 0.0, 0.06):
            for z in (-0.30, -0.26, -0.22):
                target = np.array([x, y, z])
                th = inverse_kinematics(g, target)
                if th is None:
                    continue  # outside workspace, fine
                back = forward_kinematics(g, th)
                assert back is not None, f"IK ok but FK failed for {target}"
                err = float(np.linalg.norm(back - target))
                assert err < 1e-9, f"round trip {target} -> {th} -> {back} err {err}"
                tested += 1
    print(f"IK<->FK round trip OK on {tested} reachable targets")

    print("all self-tests passed")
