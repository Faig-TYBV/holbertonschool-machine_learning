#!/usr/bin/env python3
"""
Module that trains a gensim word2vec model.
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Builds and trains a gensim word2vec model.

    Args:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences for use in training.
        window (int): Max distance between the current and predicted word.
        negative (int): Size of negative sampling.
        cbow (bool): Training type; True for CBOW, False for Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.Word2Vec: The trained model.
    """
    # sg=0 corresponds to CBOW, sg=1 corresponds to Skip-gram
    sg_value = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg_value,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
