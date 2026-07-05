#!/usr/bin/env python3
"""
Module to convert a trained Gensim Word2Vec model into a Keras Embedding layer.
"""
import keras


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a trainable keras Embedding layer.

    Parameters:
        model (gensim.models.Word2Vec): A trained gensim word2vec model.

    Returns:
        keras.layers.Embedding: A trainable Keras Embedding layer initialized
        with the word vectors from the Gensim model.
    """
    # Extract the actual word vectors (numpy array) from the Gensim model
    keyed_vectors = model.wv
    embedding_matrix = keyed_vectors.vectors
    
    # Determine vocabulary size (input_dim) and vector size (output_dim)
    vocab_size, vector_size = embedding_matrix.shape

    # Initialize the Keras Embedding layer with the extracted weights
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[embedding_matrix],
        trainable=True
    )

    return embedding_layer
