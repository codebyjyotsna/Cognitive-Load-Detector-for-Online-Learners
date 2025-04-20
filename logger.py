import logging

# Configure logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def log_message(message):
    """
    Log a message.
    """
    logging.info(message)
