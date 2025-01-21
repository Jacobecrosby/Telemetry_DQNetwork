import logging
from datetime import datetime

def setup_logger(log_directory):
    now = datetime.now()
    time = now.strftime("%H-%M-%S")
    
    # Create a logger
    logger = logging.getLogger(f"{time}")
    logger.setLevel(logging.INFO)

    # Create a file handler
    log_file = f'{log_directory}/{time}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Remove any existing handlers (to prevent duplication)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger