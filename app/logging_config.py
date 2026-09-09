"""Process-wide logging. Local DEBUG; production WARNING. No extra env vars."""

from __future__ import annotations

import logging

_NOISY = ("boto3", "botocore", "urllib3", "pymongo", "s3transfer", "matplotlib", "PIL")


def configure_logging(*, is_production: bool) -> None:
    level = logging.WARNING if is_production else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("criclab").setLevel(level)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
