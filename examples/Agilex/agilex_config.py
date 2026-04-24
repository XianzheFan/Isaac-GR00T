from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


agilex_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_joint_position",
            "left_gripper",
            "right_arm_joint_position",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "left_arm_joint_position",
            "left_gripper",
            "right_arm_joint_position",
            "right_gripper",
        ],
        action_configs=[
            # left_arm_joint_position: delta joint (matches openpi pi05 use_delta_joint_actions=True)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_gripper: absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm_joint_position: delta joint
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_gripper: absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.language.task"],
    ),
}

register_modality_config(agilex_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
