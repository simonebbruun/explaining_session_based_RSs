import pandas as pd
import tensorflow as tf


def aggregate_features(df: pd.DataFrame, session_col: str, behavior_key_col: str, seq_max_len: int) -> dict:
    
    behavior_cols = hist_cols = [behavior_key_col]
    
    dataset_dict = {}
    dataset_dict['category_' + session_col] = []
    for behavior_col in behavior_cols:
        dataset_dict['candidate_' + behavior_col] = []
        dataset_dict['behavior_' + behavior_col] = []

    for session, hist in df.groupby(session_col, sort=False):
        positive_list_dict = {}

        for hist_col in hist_cols:
            positive_list = hist[hist_col].tolist()
            positive_list_dict[hist_col] = positive_list[-(seq_max_len + 1):]

        for i in range(1, len(positive_list_dict[behavior_cols[0]])):
            dataset_dict['category_' + session_col].append(session)

            for behavior_col in behavior_cols:
                dataset_dict['candidate_' + behavior_col].append(positive_list_dict[behavior_col][i])
                dataset_dict['behavior_' + behavior_col].append(positive_list_dict[behavior_col][:i])

    return dataset_dict


def build_input(seq_max_len: int):

    input_seq = tf.keras.layers.Input(shape=(seq_max_len,), name='input_seq')
    # candidate = tf.keras.layers.Input(shape=(seq_max_len,), name='candidate')

    return input_seq


def build_embedding_layer_item(item_num: int, embedding_dim: int, l2_reg: float):

    item_embedding_layer = tf.keras.layers.Embedding(
        item_num+1, embedding_dim, name='item_embeddings', mask_zero=True,
        embeddings_regularizer=tf.keras.regularizers.L2(l2_reg))

    return item_embedding_layer


def build_embedding_layer_position(seq_max_len: int, embedding_dim: int, l2_reg: float):

    positional_embedding_layer = tf.keras.layers.Embedding(
        seq_max_len, embedding_dim, name='positional_embeddings', mask_zero=False,
        embeddings_regularizer=tf.keras.regularizers.L2(l2_reg))

    return positional_embedding_layer


def embedding_item(input_seq, item_embedding_layer):

    seq_embeddings = item_embedding_layer(input_seq)

    return seq_embeddings


def embedding_position(input_seq, positional_embedding_layer):

    positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
    positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
    positional_embeddings = positional_embedding_layer(positional_seq)

    return positional_embeddings


def layer_normalization(input_seq, epsilon = 1e-8):

    inputs_shape = input_seq.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(input_seq, [-1], keepdims=True)
    beta = tf.zeros(params_shape)
    gamma = tf.ones(params_shape)
    normalized = (input_seq - mean) / ((variance + epsilon) ** .5)
    output = gamma * normalized + beta

    return output


def multihead_attention(queries, keys, attention_dim: int, num_heads: int, dropout_rate: float):

    # Linear projections
    Q = tf.keras.layers.Dense(attention_dim, activation=None)(queries) # (N, T_q, C)
    K = tf.keras.layers.Dense(attention_dim, activation=None)(keys) # (N, T_k, C)
    V = tf.keras.layers.Dense(attention_dim, activation=None)(keys) # (N, T_k, C)

    # --- MULTI HEAD ---
    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)


    # --- SCALED DOT PRODUCT ---
    # Multiplication
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # Key Masking
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

    # Future blinding (Causality)
    diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (N, T_q, C)

    # Dropouts
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)

    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # --- MULTI HEAD ---
    # concat heads
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    # Residual connection
    outputs += queries

    return outputs


def point_wise_feed_forward(input_seq, dropout_rate: float, conv_dims: list):

    output = tf.keras.layers.Conv1D(filters=conv_dims[0], kernel_size=1, activation='relu', use_bias=True)(input_seq)
    output = tf.keras.layers.Dropout(dropout_rate)(output)

    output = tf.keras.layers.Conv1D(filters=conv_dims[1], kernel_size=1, activation=None, use_bias=True)(output)
    output = tf.keras.layers.Dropout(dropout_rate)(output)

    # Residual connection
    output += input_seq

    return output


def sasrec(item_num: int, seq_max_len: int, num_blocks: int = 2, embedding_dim: int = 100, attention_dim: int = 100,
           attention_num_heads: int = 1, conv_dims: list = [100, 100], dropout_rate: float = 0.5, l2_reg: float = 0.0):

    input_seq = build_input(seq_max_len)
    
    mask_1 = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
    
    # --- EMBEDDING LAYER ---
    item_embedding_layer = build_embedding_layer_item(item_num, embedding_dim, l2_reg)
    seq_embeddings = embedding_item(input_seq, item_embedding_layer)
    seq_embeddings *= mask_1
    
    embedding_model = tf.keras.Model(inputs=input_seq, outputs=seq_embeddings)
    
    
    model_input = tf.keras.Input(shape=(seq_max_len,embedding_dim,))
    
    mask_2 = tf.cast(tf.not_equal(model_input, 0), tf.float32)[:,:,0:1]
    
    positional_embedding_layer = build_embedding_layer_position(seq_max_len, embedding_dim, l2_reg)
    positional_embeddings = embedding_position(model_input, positional_embedding_layer)
    
    # add positional embeddings
    seq_embeddings = model_input + positional_embeddings
    
    # dropout
    seq_embeddings = tf.keras.layers.Dropout(dropout_rate)(seq_embeddings)
    
    # masking
    seq_embeddings *= mask_2
    
    # --- ATTENTION BLOCKS ---  
    seq_attention = seq_embeddings
    
    for i in range(num_blocks):
    
        # attention layer
        seq_attention = multihead_attention(queries=layer_normalization(seq_attention),
                                            keys=seq_attention,
                                            attention_dim=attention_dim,
                                            num_heads=attention_num_heads,
                                            dropout_rate=dropout_rate)
    
        # feed forward network
        seq_attention = point_wise_feed_forward(seq_attention, dropout_rate=dropout_rate, conv_dims=conv_dims)
    
        # masking
        seq_attention *= mask_2
    
    seq_attention = layer_normalization(seq_attention)[:,-1,:]
    
    output = tf.keras.layers.Dense(item_num, activation='linear')(seq_attention)
    
    model = tf.keras.Model(inputs=model_input, outputs=output)
    
    combined_model_input = tf.keras.Input(shape=(seq_max_len,))
    seq_embeddings = embedding_model(combined_model_input)
    combined_output = model(seq_embeddings)
    combined_model = tf.keras.Model(inputs=combined_model_input, outputs=combined_output)

    return combined_model
