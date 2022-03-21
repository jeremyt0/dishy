from loguru import logger
import os
import sys
import datetime

class Logger:
    # Create and set logs folder
    log_folder = os.path.join(os.path.dirname(os.getcwd()), "logs")
    # os.makedirs(log_folder, exist_ok=True)

    # Create filename
    log_file = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

    # Create log file
    log_filepath = os.path.join(log_folder, log_file)

    # Remove default logger
    logger.remove()
    # Add message only in stdout
    logger.add(sys.stdout, format="{message}")
    # Add log file to logger
    logger.add(log_filepath, enqueue=True, backtrace=True, diagnose=True)

# Exported log variable
LOGGER = logger
