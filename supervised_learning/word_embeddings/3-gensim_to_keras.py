#!/usr/bin/env python3
"""
Module to convert a trained Gensim Word2Vec model into a Keras Embedding layer.
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a trainable keras Embedding layer.

    Parameters:
        model (gensim.models.Word2Vec): A trained gensim word2vec model.

    Returns:
        tf.keras.layers.Embedding: A trainable Keras Embedding layer initialized
        with the word vectors from the Gensim model.
    """
    # Extract the exact word vectors (numpy array) from the model
    embedding_matrix = model.wv.vectors
    
    # Determine vocabulary size and vector size directly from the matrix
    vocab_size, vector_size = embedding_matrix.shape

    # Return the initialized Embedding layer
    return tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[embedding_matrix],
        trainable=True
    )
