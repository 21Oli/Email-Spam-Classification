"""
Text Preprocessing Utilities


This module contains all text preprocessing
functions used throughout the application.
"""

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# Download NLTK Resources
# --------------------------------------------------

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# --------------------------------------------------
# Initialize NLP Objects
# --------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()


# --------------------------------------------------
# Text Cleaning Function
# --------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess email text.

    Steps
    -----
    1. Lowercase
    2. Remove HTML
    3. Remove URLs
    4. Remove Email Addresses
    5. Remove Numbers
    6. Remove Punctuation
    7. Remove Extra Spaces
    8. Remove Stopwords
    9. Lemmatize

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned email.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove HTML

    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs

    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove Email Addresses

    text = re.sub(r"\S+@\S+", " ", text)

    # Remove Numbers

    text = re.sub(r"\d+", " ", text)

    # Remove Punctuation

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove Extra Spaces

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [

        LEMMATIZER.lemmatize(word)

        for word in words

        if word not in STOP_WORDS

    ]

    return " ".join(words)


# --------------------------------------------------
# Optional Utility
# --------------------------------------------------

def preprocess_dataframe(df, column):
    """
    Apply preprocessing to
    an entire dataframe column.
    """

    df = df.copy()

    df[column] = df[column].apply(preprocess_text)

    return df