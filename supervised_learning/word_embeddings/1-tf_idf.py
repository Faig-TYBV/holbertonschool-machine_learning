#!/usr/bin/env python3
"""
Module that contains the tf_idf function for NLP feature extraction.
"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list): A list of vocabulary words to use. If None,
            all unique words within sentences will be used.

    Returns:
        tuple:
            numpy.ndarray: TF-IDF embeddings matrix of shape (s, f).
            numpy.ndarray: The features used for the embeddings.
    """
    extracted_sentences = []
    
    # 1. Preprocess sentences (matching bag_of_words logic)
    for sentence in sentences:
        clean_sentence = re.sub(r'[^a-z0-9]', ' ', sentence.lower())
        words = [w for w in clean_sentence.split() if len(w) > 1]
        extracted_sentences.append(words)

    # 2. Build Vocabulary (Features)
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

    # 3. Calculate Term Frequency (TF - raw count)
    for i, words in enumerate(extracted_sentences):
        for word in words:
            if word in feature_dict:
                embeddings[i, feature_dict[word]] += 1

    # 4. Calculate Document Frequency (DF)
    df = np.zeros(f, dtype=int)
    for words in extracted_sentences:
        unique_words = set(words)
        for word in unique_words:
            if word in feature_dict:
                df[feature_dict[word]] += 1

    # 5. Calculate Inverse Document Frequency (IDF)
    # Using standard formula: log(N / df)
    idf = np.zeros(f, dtype=float)
    for i in range(f):
        if df[i] > 0:
            idf[i] = np.log(s / df[i])
        else:
            idf[i] = 0.0  # Fallback if a provided vocab word is never used

    # 6. Compute TF-IDF
    for i in range(s):
        embeddings[i, :] = embeddings[i, :] * idf

    return embeddings, np.array(features)
