#!/usr/bin/env python3
"""
Module for creating, building, and training a Word2Vec model using Gensim.
"""
import gensim


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
        gensim.models.Word2Vec: The trained Gensim Word2Vec model.
    """
    sg_mode = 0 if cbow else 1

    # 1. Create the model (without automatically building/training)
    try:
        # Gensim 4.x expected parameters
        model = gensim.models.Word2Vec(
            vector_size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=sg_mode,
            seed=seed,
            workers=workers
        )
    except TypeError:
        # Fallback for Gensim 3.x
        model = gensim.models.Word2Vec(
            size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=sg_mode,
            seed=seed,
            workers=workers
        )

    # 2. Build the vocabulary
    model.build_vocab(sentences)

    # 3. Train the model
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
