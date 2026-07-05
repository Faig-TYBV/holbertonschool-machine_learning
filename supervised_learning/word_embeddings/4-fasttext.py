#!/usr/bin/env python3
"""
Module for creating, building, and training a FastText model using Gensim.
"""

import gensim

def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim fastText model.
    
    Parameters:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences of a word for use in training.
        negative (int): Size of negative sampling.
        window (int): Maximum distance between the current and predicted word within a sentence.
        cbow (bool): True for CBOW; False for Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.
        
    Returns:
        gensim.models.fasttext.FastText: The trained FastText model.
    """
    # Map the 'cbow' boolean to Gensim's 'sg' (skip-gram) parameter. 
    # sg=0 means CBOW, sg=1 means Skip-gram.
    sg_param = 0 if cbow else 1
    
    # Initialize, build vocabulary, and train the model 
    # (Passing 'sentences' directly handles building and training automatically)
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=sg_param,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    
    return model
