import json
import enum

path = "/ai/src/ai_detection/ai_detection/{}"


class Camera(enum.Enum):
    DRIVE = 0
    ARM = 1


def get_path(subpath: str) -> str:
    return path.format(subpath)


def get_config() -> dict:
    config = {}
    path = get_path("config.json")

    with open(path, "r") as f:
        config = json.load(f)

    return config
