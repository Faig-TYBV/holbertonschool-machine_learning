#!/usr/bin/env python3
"""
Module that contains the tf_idf function for NLP feature extraction.
"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix with L2 normalization.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list): A list of vocabulary words to use.

    Returns:
        tuple: (embeddings, features)
    """
    extracted_sentences = []
    for sentence in sentences:
        clean_sentence = re.sub(r'[^a-z0-9]', ' ', sentence.lower())
        words = [w for w in clean_sentence.split() if len(w) > 1]
        extracted_sentences.append(words)

    if vocab is None:
        features_set = set()
        for words in extracted_sentences:
            features_set.update(words)
        features = sorted(list(features_set))
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=float)
    feature_dict = {feature: idx for idx, feature in enumerate(features)}

    # TF: Raw counts
    for i, words in enumerate(extracted_sentences):
        for word in words:
            if word in feature_dict:
                embeddings[i, feature_dict[word]] += 1

    # Smooth IDF: log((1 + N) / (1 + df)) + 1
    df = np.zeros(f, dtype=int)
    for words in extracted_sentences:
        unique_words = set(words)
        for word in unique_words:
            if word in feature_dict:
                df[feature_dict[word]] += 1

    idf = np.log((1 + s) / (1 + df)) + 1
    embeddings = embeddings * idf

    # L2 Normalization: row / sqrt(sum(row^2))
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    embeddings = np.divide(embeddings, norm, out=np.zeros_like(embeddings),
                           where=norm != 0)

    return embeddings, np.array(features)
