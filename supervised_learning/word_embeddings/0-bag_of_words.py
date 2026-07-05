#!/usr/bin/env python3
"""
Module that contains the bag_of_words function for NLP feature extraction.
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list): A list of vocabulary words to use. If None,
            all unique words within sentences will be used.

    Returns:
        tuple:
            numpy.ndarray: Embeddings matrix of shape (s, f).
            numpy.ndarray: The features used for the embeddings.
    """
    extracted_sentences = []
    for sentence in sentences:
        # Lowercase and replace non-alphanumeric characters with spaces
        clean_sentence = re.sub(r'[^a-z0-9]', ' ', sentence.lower())
        # Split into words and keep only those with length > 1
        words = [w for w in clean_sentence.split() if len(w) > 1]
        extracted_sentences.append(words)

    if vocab is None:
        # Extract unique words and sort alphabetically
        features_set = set()
        for words in extracted_sentences:
            features_set.update(words)
        features = sorted(list(features_set))
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Dictionary for faster O(1) feature index lookup
    feature_dict = {feature: idx for idx, feature in enumerate(features)}

    for i, words in enumerate(extracted_sentences):
        for word in words:
            if word in feature_dict:
                embeddings[i, feature_dict[word]] += 1

    return embeddings, np.array(features)
