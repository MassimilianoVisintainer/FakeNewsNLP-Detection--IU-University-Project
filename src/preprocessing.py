"""
preprocessing.py
----------------
Module for text preprocessing in the Fake News Detection project.

This module provides:
- A configuration object to define cleaning rules.
- A TextPreprocessor class with methods for cleaning, tokenization, 
  lemmatization, stopword removal, and reconstruction.
"""

from dataclasses import dataclass
from typing import List, Union
import html
import re
import contractions 
import spacy

@dataclass
class PreprocessConfig:
    """
    Configuration settings for preprocessing.
    Adjust these flags to control the cleaning pipeline.
    """
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = True
    expand_contractions: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    keep_negations: bool = True
    remove_bylines: bool = True
    remove_html: bool = True


class TextPreprocessor:
    """
    A reusable text preprocessing pipeline.

    Methods:
        clean_basic(text): remove HTML, URLs, emails, bylines, punctuation, etc.
        normalize_contractions(text): expand contractions ("don't" -> "do not").
        tokenize(text): split text into tokens (using spaCy or NLTK).
        lemmatize(tokens): lemmatize words to their base form.
        remove_stopwords(tokens): drop common stopwords (keep negations if set).
        postprocess(tokens): drop short tokens, join back into a string.
        transform(text): full pipeline on a single string.
        transform_corpus(texts): full pipeline on a list of texts.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        # Load spaCy English model for tokenization & lemmatization
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def clean_basic(self, text: str) -> str:
        """
        Fully general cleaning of news text:
        - lowercase
        - unescape HTML
        - remove HTML tags
        - remove URLs
        - remove emails
        - remove leading bracketed notes (boilerplate)
        - remove datelines and source prefixes
        - collapse multiple spaces
        """

        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")

        # 1. Lowercase
        if self.config.lowercase:
            text = text.lower()

        # 2. Unescape HTML entities
        text = html.unescape(text)

        # 3. Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # 4. Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)

        # 5. Remove emails
        if self.config.remove_emails:
            text = re.sub(r'\S+@\S+\.\S+', '', text)

        # 6. Remove leading bracketed notes (boilerplate)
        # e.g., (FILE PHOTO), (CNN), (Reuters Photographer)
        text = re.sub(r'^\s*\([^\)]+\)\s*', '', text)

        # 7. Remove datelines: CITY (SOURCE) -
        text = re.sub(r'^[A-Z\s]+\(.*?\)\s*[-–—]\s*', '', text)

        # 8. Remove source prefixes: Reuters -, CNN —, etc.
        text = re.sub(r'^(Reuters|CNN|AP|AFP|BBC|Fox News)\s*[-–—]\s*', '', text, flags=re.IGNORECASE)

        # 9. Collapse multiple spaces and strip

        # Remove punctuation except apostrophes and hyphens inside words
        text = re.sub(r"[^\w\s'-]", '', text)

        #  Collapse multiple spaces and strip
        text = re.sub(r'\s+', ' ', text).strip()

        return text


        

    def normalize_contractions(self, text: str) -> str:
        """Expand common contractions (e.g., don't → do not)."""
        if not isinstance(text, str):
            return text
        if self.config.expand_contractions:
         # contractions.fix handles a lot of cases (requires package installed)
            return contractions.fix(text)
        return text


    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens (words) using spaCy.
        Preserves internal hyphens and apostrophes.
        """

        doc = self.nlp(text)
        tokens = [token.text for token in doc]

        return tokens

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form using spaCy.
        Assumes tokens are a list of strings from tokenize().
        """
        text = ' '.join(tokens)
        doc = self.nlp(text)

        lemmas = [token.lemma_ for token in doc]

        return lemmas

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens using spaCy's stopword list.
        Keeps negations if `keep_negations` is True.
        """
        # Initialize filtered tokens list
        filtered_tokens = []

        # Iterate trough each token
        for token in tokens:
            # Check if the token is a stop word
            if token in self.nlp.Defaults.stop_words:
                # Check if keep_negations is enabled and the token is a negation
                if self.config.keep_negations and token in ['no', 'not', 'nor']:
                    filtered_tokens.append(token)
            else:
                filtered_tokens.append(token)
        
        return filtered_tokens

    def postprocess(self, tokens: List[str]) -> str:
        """
        Clean up token list:
        - Remove very short tokens (e.g., length < 2)
        - Remove pure punctuation
        - Join tokens back into a single string
        """

        processed = [
            token for token in tokens
            if len(token) > 1 and any(c.isalnum() for c in token)
        ]

        return " ".join(processed)


    def transform(self, text: str) -> str:
        """
        Run the full preprocessing pipeline on a single string:
        - clean_basic
        - normalize_contractions
        - tokenize
        - lemmatize
        - remove_stopwords
        - postprocess
        """
        
        # General cleaning of the text
        text = self.clean_basic(text)

        # Normalization of the text
        text = self.normalize_contractions(text)

        # Splitting text into tokens
        tokens = self.tokenize(text)

        # Lemmatize if needed
        if self.config.lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Remove stopwords if needed
        if self.config.remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Post process the text
        text_clean = self.postprocess(tokens)

        return text_clean



    def transform_corpus(self, texts: List[str]) -> List[str]:
        """
            Apply full preprocessing pipeline to a list of strings (e.g., dataset column).
        """

        return [self.transform(text) for text in texts]


    def transform_corpus_fast(self, texts: list[str], batch_size: int = 200, n_process: int = 1) -> list[str]:
        """
        Faster preprocessing of a list of texts using spaCy's nlp.pipe().
        Uses smaller batch size to reduce memory usage.
        """
        results = []

        for doc in self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            tokens = [t.lemma_ if self.config.lemmatize else t.text for t in doc]

            # stopword filtering
            if self.config.remove_stopwords:
                tokens = [
                    t for t in tokens
                    if not (t in self.nlp.Defaults.stop_words and not (self.config.keep_negations and t in ["no", "not", "nor"]))
                ]

            # remove very short / non-alphanumeric tokens
            tokens = [t for t in tokens if len(t) > 1 and any(c.isalnum() for c in t)]

            results.append(" ".join(tokens))

        return results


