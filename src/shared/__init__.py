"""CLF-coupled distributed controller for a mobile manipulator.

The base and arm solve local velocity-control subproblems in parallel. ADMM
coordinates only their scalar contributions to the derivative of a common
Control Lyapunov Function (CLF):

    s_b = a_b.T @ u_b
    s_a = a_a.T @ u_a
    s_b + s_a <= b_clf

No Cartesian position-consensus variable from the former controller remains.
"""

from .base_subsystem import BaseSubsystem
from .arm_subsystem import ArmSubsystem
from .coordinator import (
    Mailbox,
    LatestValueChannel,
    BaseMCU,
    ArmMCU,
    DistributedADMMCoordinator,
)

__all__ = [
    "BaseSubsystem",
    "ArmSubsystem",
    "Mailbox",
    "LatestValueChannel",
    "BaseMCU",
    "ArmMCU",
    "DistributedADMMCoordinator",
]
