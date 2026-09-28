"""Monkey-patch LIBERO to use ECO65 + RethinkGripper.

Import and call `patch()` before creating any LIBERO environment.
"""

import logging

logger = logging.getLogger(__name__)

_IS_PATCHED = False


def patch():
    """Monkey-patch robosuite SingleArmEnv to force ECO65 + RethinkGripper.

    Any subclass (including LIBERO's OffScreenRenderEnv) will automatically
    use ECO65 instead of Panda after this patch is applied.
    """
    global _IS_PATCHED
    if _IS_PATCHED:
        return

    import robosuite.environments.manipulation.single_arm_env as _sae

    _original_init = _sae.SingleArmEnv.__init__

    def _patched_init(self, robots=None, gripper_types=None, **kwargs):
        return _original_init(
            self,
            robots=["ECO65"],
            gripper_types="RethinkGripper",
            **kwargs,
        )

    _sae.SingleArmEnv.__init__ = _patched_init
    _IS_PATCHED = True
    logger.info("ECO65 patch applied: SingleArmEnv now forces ECO65 + RethinkGripper")
