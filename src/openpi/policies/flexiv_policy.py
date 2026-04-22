"""Policy I/O transforms for Flexiv LeRobot datasets (e.g. ``datasets/stack_cubes_30``).

Dataset features (see ``datasets/stack_cubes_30/meta/info.json``):
  - ``observation.state`` / ``action``: 7-D (main_x..rz, main_gripper)
  - ``observation.images.primary``, ``observation.images.wrist``

These transforms expect **repacked** keys aligned with the Libero-style inference convention
(``observation/image``, ``observation/wrist_image``, ``observation/state``). Map dataset
fields with :class:`openpi.transforms.RepackTransform`, for example (paths depend on how
your loader flattens nested features; adjust source strings if needed)::

    RepackTransform(
        {
            \"observation/image\": \"observation/images/primary\",
            \"observation/wrist_image\": \"observation/images/wrist\",
            \"observation/state\": \"observation/state\",
            \"actions\": \"action\",
            \"prompt\": \"prompt\",
        }
    )
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_flexiv_example() -> dict:
    """Random input matching the repacked keys used by :class:`FlexivInputs`."""
    return {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "stack the red cube on the green cube",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FlexivInputs(transforms.DataTransformFn):
    """Map repacked Flexiv / stack-cubes observations into the model observation format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FlexivOutputs(transforms.DataTransformFn):
    """Map model actions back to the dataset action layout (7-D pose + gripper)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7], dtype=np.float32)}
