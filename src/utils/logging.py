import logging
import sys


def setup_logging(level=logging.INFO):
    fmt = "%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout)
