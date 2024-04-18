import json
import enum

path = "/ai/src/ai_detection/ai_detection/{}"


class Camera(enum.Enum):
    DRIVE = "/dev/v4l/by-id/usb-046d_Logitech_BRIO_497E5D01-video-index0"
    ARM = "/dev/arm_camera"


def get_path(subpath: str) -> str:
    return path.format(subpath)


def get_config() -> dict:
    config = {}
    path = get_path("config.json")

    with open(path, "r") as f:
        config = json.load(f)

    return config
