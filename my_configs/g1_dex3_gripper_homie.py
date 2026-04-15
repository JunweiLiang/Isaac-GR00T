from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
from gr00t.data.embodiment_tags import EmbodimentTag

g1_wbc_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high"],
    ),

    # STATE: We feed the model EVERYTHING so it understands full body posture.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arms", "hands", "waist", "legs", # "triggers", "loco_cmd"
        ],
        sin_cos_embedding_keys=["arms", "hands", "waist", "legs"],
    ),

    # ACTION: We restrict the model to ONLY output what you requested.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "arms",
            "triggers",
            "loco_cmd"
        ],
        action_configs=[
            # Arms (14D): Smooth relative joint control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),

            # Triggers (2D): Absolute target state (0.0 to 1.0)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),

            # Loco Commands (4D): Absolute target velocities
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),

    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_wbc_config, embodiment_tag=EmbodimentTag.G1_DEX3_HOMIE)
