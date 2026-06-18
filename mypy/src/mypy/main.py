from .utils import setup_logging
import logging
import sys


def main() -> None:
    logger = logging.getLogger(__name__)
    print("This is a print statement, not a log message.")
    print(f"Name of the logger: {logger.name}")
    # is this root logger
    print(f"Is this the root logger? {logger is logging.getLogger()}")


if __name__ == "__main__":
    setup_logging("configs/logging.yaml")  # configure ONCE, here
    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")

    from .cost_ladder import main as cost_ladder_main

    cost_ladder_main()

    sys.exit()
