import pandas as pd
import numpy as np
import torch
import holidays
import datetime

# --- 1. Region and Holiday Functions ---

def _region_to_country_code(r):
    """Maps simple region strings to holiday country codes."""
    if pd.isna(r):
        return 'US'
    s = str(r).upper()
    if 'CAN' in s or s == 'CA':
        return 'CA'
    if 'USA' in s or s == 'US':
        return 'US'
    # default to US if unknown
    return 'US'

def _get_holiday_calendars(country_codes, years=None):
    """
    Builds a dictionary of holiday calendars from a list of country codes.
    
    Args:
    - country_codes (list or pd.Series): A list of country codes (e.g., ['US', 'CA']).
    - years (list): List of years to generate holidays for.
    """
    if years is None:
        years = [datetime.date.today().year]
    
    holiday_calendars = {}
    for c in set(country_codes): # Use set for unique codes
        if c not in holiday_calendars:
            try:
                holiday_calendars[c] = holidays.CountryHoliday(c, years=years)
            except Exception:
                holiday_calendars[c] = None
    return holiday_calendars

def _is_holiday(row, holiday_calendars, day_col='Day', country_col='_country_code'):
    """
    Helper function for .apply() to check if a row's date is a holiday.
    
    Expects:
    - row (pd.Series): A pandas DataFrame row.
    - holiday_calendars (dict): The dictionary from _get_holiday_calendars.
    - day_col (str): The name of the column containing the date.
    - country_col (str): The name of the column with the country code.
    """
    cal = holiday_calendars.get(row[country_col])
    if cal is None:
        return 0 # Not a holiday if calendar is missing
    
    date_val = row[day_col]
    
    # Try to get .date() if it's a datetime object
    try:
        date_obj = date_val.date()
    except AttributeError:
        # Assume it's already a date object
        date_obj = date_val
    except Exception:
        # If it's not a date-like object at all
        return 0
    
    return 1 if date_obj in cal else 0

# --- 2. Data Cleaning Functions ---

def clean_currency(x):
    """Converts currency strings to float by removing '$', ',', and whitespace."""
    return float(x.replace('$', '').replace(',', '').strip()) if isinstance(x, str) else x

def convert_percent_to_float(value):
    """Converts percentage strings to float decimals."""
    # Pass through NaNs
    if pd.isna(value):
        return np.nan
    
    value = str(value).strip()
    
    # Handle '∞' symbol (infinite change)
    if value == '∞':
        return np.inf
    
    # Try to remove '%' and convert to float
    try:
        # Remove '%' and strip any whitespace
        cleaned_value = value.replace('%', '').strip()
        # Convert to float
        float_value = float(cleaned_value)
        # Convert to decimal (e.g., 900% -> 9.0, 0% -> 0.0)
        return float_value / 100.0
    except ValueError:
        # In case there are other non-numeric values
        return np.nan

def calculate_days_to_next(d, course_start_dts):
    """
    Calculate days from date d to the next course start date.
    
    Args:
    - d (datetime): The date to calculate from.
    - course_start_dts (list): List of ISO date strings for course start dates.
    
    Returns:
    - int or nan: Days to next course start, or nan if no future course.
    """
    starts = pd.to_datetime(course_start_dts).sort_values()
    diffs = [(int((cs - d).days)) for cs in starts if (cs - d).days >= 0]
    return int(min(diffs)) if diffs else np.nan

# --- 3. BERT Embedding Function ---

def get_bert_embedding(text_list, model, tokenizer, device):
    """
    Generates BERT embeddings (CLS token) for a list of texts.
    (Uses 'transformers' library)
    """
    if text_list is None or len(text_list) == 0:
        return np.array([])
        
    # Ensure it's a list, not a Series
    if not isinstance(text_list, list):
        text_list = text_list.tolist()
        
    # Tokenize: Add special tokens ([CLS], [SEP]), pad/truncate to max length
    encoded_input = tokenizer(
        text_list, 
        padding=True, 
        truncation=True, 
        return_tensors='pt', 
        max_length=64 # Keywords are short
    )
    
    # Move inputs to the same device as the model
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Generate embeddings (no gradient needed for inference)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Extract the embedding of the [CLS] token (first token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    
    return cls_embeddings.cpu().numpy()