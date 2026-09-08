"""ADRC-flavored persistent bias corrector for Cartesian tracking error.

Scoped for PP-mode motors where the host can only bias the commanded position
reference -- no torque/velocity/gain access. Learns a per-axis (X/Y/Z, no
cross-coupling model) additive bias between commanded and FK-observed actual
position, using move_xyz()'s existing CAN feedback. In-memory only -- resets
on process restart. Inert unless config.ADRC_BIAS_ENABLE is True.
"""
from delta_common import config


class BiasObserver:
    """Per-axis persistent bias estimate d_hat.

    Model: actual = commanded + b, where b is an unknown, slowly-varying
    disturbance (backlash, cross-arm coupling, gravity sag, kinematics
    residual). Callers command (target - d_hat); the observed residual
    e = actual - target = b - d_hat drives d_hat toward b over repeated
    calls via a leaky integrator.
    """

    def __init__(self):
        self.d_hat = [0.0, 0.0, 0.0]  # x, y, z mm

    def correction(self):
        """(dx, dy, dz) to SUBTRACT from the desired target before IK."""
        return tuple(self.d_hat)

    def update(self, ex: float, ey: float, ez: float) -> bool:
        """Update d_hat from signed per-axis residual e = actual - target (mm).

        Per-axis deadband (config.ADRC_BIAS_MIN_ERR_MM) and clamp
        (config.ADRC_BIAS_MAX_MM) applied. Returns True if any axis changed.
        """
        beta, max_mm, min_err = (
            config.ADRC_BIAS_BETA, config.ADRC_BIAS_MAX_MM, config.ADRC_BIAS_MIN_ERR_MM
        )
        changed = False
        for i, e in enumerate((ex, ey, ez)):
            if abs(e) < min_err:
                continue
            d = max(-max_mm, min(max_mm, self.d_hat[i] + beta * e))
            if d != self.d_hat[i]:
                self.d_hat[i] = d
                changed = True
        if changed:
            print(
                f"[ADRC bias] d_hat=({self.d_hat[0]:+.3f},{self.d_hat[1]:+.3f},"
                f"{self.d_hat[2]:+.3f})mm  e=({ex:+.2f},{ey:+.2f},{ez:+.2f})mm"
            )
        return changed

    def reset(self):
        self.d_hat = [0.0, 0.0, 0.0]
