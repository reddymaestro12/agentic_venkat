import logging


def get_weather(city: str) -> str:
    """
    Get weather information for a given city.

    Args:
        city (str): Name of the city.

    Returns:
        str: Weather information.
    """
    try:
        if not city:
            raise ValueError("City name cannot be empty.")

        city = city.strip()

        if not city:
            raise ValueError("City name cannot contain only whitespace.")

        return f"It's always sunny in {city}!"

    except ValueError as exc:
        logging.warning(f"Invalid city input: {exc}")
        return str(exc)

    except Exception as exc:
        logging.exception("Unexpected error while retrieving weather information.")
        return f"An unexpected error occurred: {exc}"
