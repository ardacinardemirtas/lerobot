import importlib
import sys
import types

import numpy as np
import pytest


try:
    importlib.import_module("ikpy.chain")
except ModuleNotFoundError:
    ikpy_module = types.ModuleType("ikpy")
    chain_module = types.ModuleType("ikpy.chain")
    chain_module.Chain = type("Chain", (), {})
    ikpy_module.chain = chain_module
    sys.modules.setdefault("ikpy", ikpy_module)
    sys.modules.setdefault("ikpy.chain", chain_module)


QP_MODULES = (
    importlib.import_module("move_to_position_qp"),
    importlib.import_module("move_to_position_qp_hold_orient"),
)


def _empty_q(module) -> np.ndarray:
    return np.zeros(len(module.MOTOR_NAMES), dtype=float)


def _active_values(module, q: np.ndarray) -> np.ndarray:
    return np.asarray([q[module.JOINT_INDEX[name]] for name in module.ARM_JOINTS], dtype=float)


@pytest.mark.parametrize("module", QP_MODULES, ids=[module.__name__ for module in QP_MODULES])
def test_integrate_command_uses_measured_dt_and_preserves_non_active_joints(module):
    q_cmd = _empty_q(module)
    q_actual = _empty_q(module)
    q_cmd[module.JOINT_INDEX["wrist_roll"]] = 12.0
    q_cmd[module.JOINT_INDEX["gripper"]] = 33.0
    q_actual[module.JOINT_INDEX["wrist_roll"]] = -90.0
    q_actual[module.JOINT_INDEX["gripper"]] = 0.0

    qdot = np.deg2rad(np.asarray([50.0, -25.0, 0.0, 10.0]))

    out = module._integrate_command(q_cmd, q_actual, qdot, module.ARM_JOINTS, 0.02)

    np.testing.assert_allclose(_active_values(module, out), [1.0, -0.5, 0.0, 0.2], atol=1e-9)
    assert out[module.JOINT_INDEX["wrist_roll"]] == q_cmd[module.JOINT_INDEX["wrist_roll"]]
    assert out[module.JOINT_INDEX["gripper"]] == q_cmd[module.JOINT_INDEX["gripper"]]


@pytest.mark.parametrize("module", QP_MODULES, ids=[module.__name__ for module in QP_MODULES])
def test_integrate_command_clamps_command_ahead(module):
    q_cmd = _empty_q(module)
    q_actual = _empty_q(module)
    qdot = np.deg2rad(np.asarray([1000.0, -1000.0, 2000.0, -2000.0]))

    out = module._integrate_command(q_cmd, q_actual, qdot, module.ARM_JOINTS, 0.02)

    np.testing.assert_allclose(
        _active_values(module, out),
        [
            module.MAX_CMD_AHEAD_DEG,
            -module.MAX_CMD_AHEAD_DEG,
            module.MAX_CMD_AHEAD_DEG,
            -module.MAX_CMD_AHEAD_DEG,
        ],
        atol=1e-9,
    )


@pytest.mark.parametrize("module", QP_MODULES, ids=[module.__name__ for module in QP_MODULES])
def test_integrate_command_caps_large_delayed_dt(module):
    q_cmd = _empty_q(module)
    q_actual = _empty_q(module)
    qdot = np.deg2rad(np.asarray([20.0, -20.0, 0.0, 10.0]))

    out = module._integrate_command(q_cmd, q_actual, qdot, module.ARM_JOINTS, 1.0)

    np.testing.assert_allclose(_active_values(module, out), [1.0, -1.0, 0.0, 0.5], atol=1e-9)


@pytest.mark.parametrize("module", QP_MODULES, ids=[module.__name__ for module in QP_MODULES])
def test_integrate_command_keeps_command_near_measured_position(module):
    q_cmd = _empty_q(module)
    q_actual = _empty_q(module)
    for name in module.ARM_JOINTS:
        q_cmd[module.JOINT_INDEX[name]] = 10.0
        q_actual[module.JOINT_INDEX[name]] = 20.0
    qdot = np.zeros(len(module.ARM_JOINTS), dtype=float)

    out = module._integrate_command(q_cmd, q_actual, qdot, module.ARM_JOINTS, 0.02)

    np.testing.assert_allclose(_active_values(module, out), [16.0, 16.0, 16.0, 16.0], atol=1e-9)
