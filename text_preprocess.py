# Library for some simple text preprocessing steps
import re

def clean_text(t):
    # Removes all special characters from text
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', t)
    return text
