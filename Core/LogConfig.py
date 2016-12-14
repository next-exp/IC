"""
Defines a logger (works also in the NB)
JJGC August 2016
"""
import logging
import sys

logger = logging.getLogger()

# create console handler
ch = logging.StreamHandler()
logger.addHandler(ch)
logger.handlers[0].stream = sys.stdout
logger.setLevel(logging.INFO)
