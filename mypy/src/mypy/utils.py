from __future__ import annotations
from typing import Any, Mapping
from pathlib import Path
import logging
import logging.config

from omegaconf import OmegaConf


# Single shot setup for Python logging module.
def setup_logging(
    config_path: str | Path = "config/logging.yaml",
    overrides: Mapping[str, Any] | None = None,
) -> logging.Logger:
    """
    Load logging config with OmegaConf and apply it via dictConfig -- once.

    Example usage:
        def main() -> None:
            setup_logging(
                    overrides={
                        "root": {
                            "handlers": ["stderr", "file-stderr"]
                        }
                    }
            )
    """

    cfg = OmegaConf.load(config_path)

    if overrides is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(overrides)))

    # dictconfig wants a plain dict; resove ${oc.env:} now.
    cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

    # Ensure parent dirs exist for every file-based handler.
    for handler in cfg_dict.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(cfg_dict)
    return logging.getLogger(__name__)
