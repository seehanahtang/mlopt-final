"""
DEPRECATED: Legacy helpers.py - Use utils module directly instead.

This file maintains backward compatibility. New code should import from:
  - from utils.data_cleaning import clean_currency, convert_percent_to_float
  - from utils.date_features import _region_to_country_code, _is_holiday, etc.
  - from utils.embeddings import get_tfidf_embeddings, get_bert_embeddings_pipeline, etc.

Or simply:
  - from utils import clean_currency, convert_percent_to_float, ...
"""

# Re-export everything from utils for backward compatibility
from utils import (
    clean_currency,
    convert_percent_to_float,
    _region_to_country_code,
    _get_holiday_calendars,
    _is_holiday,
    calculate_days_to_next,
    get_tfidf_embeddings,
    get_bert_embeddings_pipeline,
    get_bert_embedding,
)

__all__ = [
    'clean_currency',
    'convert_percent_to_float',
    '_region_to_country_code',
    '_get_holiday_calendars',
    '_is_holiday',
    'calculate_days_to_next',
    'get_tfidf_embeddings',
    'get_bert_embeddings_pipeline',
    'get_bert_embedding',
]