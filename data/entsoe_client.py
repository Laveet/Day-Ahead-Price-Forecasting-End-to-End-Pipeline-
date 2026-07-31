"""
entsoe_client.py

Single shared ENTSO-E API client used by every data-download module, so
the API key is loaded once instead of being duplicated in every file.
"""

import os
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()
_api_key = os.getenv("ENTSOE_API_KEY")


def get_client() -> EntsoePandasClient:
    if not _api_key:
        raise RuntimeError(
            "ENTSOE_API_KEY not found. Add it to your .env file at the project root."
        )
    return EntsoePandasClient(api_key=_api_key)
