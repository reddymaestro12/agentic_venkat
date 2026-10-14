from dotenv import load_dotenv, find_dotenv
import logging


def load_config() -> bool:
    """
    Load environment variables from a .env file with validation.

    Returns:
        bool: True if the .env file is loaded successfully, False otherwise.
    """
    try:
        env_file = find_dotenv()

        if not env_file:
            raise FileNotFoundError(".env file not found.")

        load_dotenv(env_file)
        logging.info(f"Loaded environment variables from: {env_file}")
        return True

    except FileNotFoundError as err:
        logging.warning(str(err))
        return False

    except Exception as exc:
        logging.exception("Unexpected error while loading configuration.")
        return False