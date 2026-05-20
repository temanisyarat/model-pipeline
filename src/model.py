import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), name='attention_w')
        self.b = self.add_weight(shape=(1,), name='attention_b')

    def call(self, x):
        score = tf.matmul(x, self.W) + self.b
        alpha = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(x * alpha, axis=1)
        return context


def build_mobile_sign_gru(input_dim, num_classes, max_len, hidden_dim=192, num_layers=2,
                          dropout=0.3, bidirectional=True, l2_reg=1e-3,
                          conv_filters=None, conv_kernel_size=5,
                          spatial_dropout=0.2, recurrent_dropout=0.2,
                          use_mask_concat=True):
    if conv_filters is None:
        conv_filters = [128, 128]
    feat_dim = input_dim // 2 if use_mask_concat else input_dim

    inputs = layers.Input(shape=(max_len, input_dim), name='input')

    if use_mask_concat:
        feat = inputs[..., :feat_dim]
        mask = inputs[..., feat_dim:]
        x = feat * mask
    else:
        x = inputs

    x = layers.SpatialDropout1D(spatial_dropout)(x)

    for i, filters in enumerate(conv_filters):
        x = layers.Conv1D(filters, conv_kernel_size, padding='same',
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name=f'conv1d_{i}')(x)
        x = layers.BatchNormalization(name=f'conv_bn_{i}')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2, name=f'pool_{i}')(x)

    for i in range(num_layers):
        gru = layers.GRU(
            hidden_dim,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=True,
            name=f'gru_{i}'
        )
        if bidirectional:
            x = layers.Bidirectional(gru, name=f'bigru_{i}')(x)
        else:
            x = gru(x)

    x = TemporalAttention(name='temporal_attention')(x)

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_dim, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation=None, name='output')(x)

    return models.Model(inputs, outputs)
