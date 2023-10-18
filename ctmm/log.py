import logging

# format
log_format = "[%(asctime)s - %(levelname)s] %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
fmt = logging.Formatter(fmt=log_format, datefmt=date_format)

logging.basicConfig(format=log_format, datefmt=date_format, level=logging.INFO)
logger = logging.getLogger('CTMM')

