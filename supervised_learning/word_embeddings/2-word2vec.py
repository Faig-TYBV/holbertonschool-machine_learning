#!/usr/bin/env python3
"""
Module for creating, building, and training a Word2Vec model using Gensim.
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim Word2Vec model.

    Parameters:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum frequency count of a word to be included.
        window (int): Maximum distance between current and predicted word.
        negative (int): Number of negative samples to use.
        cbow (bool): Determines training type (True for CBOW, False for Skip-gram).
        epochs (int): Number of iterations over the corpus.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.

    Returns:
        Word2Vec: The trained Gensim Word2Vec model.
    """
    # Gensim uses parameter 'sg' where 0 = CBOW and 1 = Skip-gram
    sg_mode = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg_mode,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
