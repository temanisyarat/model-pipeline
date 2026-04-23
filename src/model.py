from tensorflow.keras import layers, models


def build_mobile_sign_gru(input_dim, num_classes, max_len, hidden_dim=192, num_layers=2, dropout=0.25, bidirectional=True):
    """TensorFlow/Keras implementation of MobileSignGRU with explicit shapes."""
    inputs = layers.Input(shape=(max_len, input_dim), name='input')
    x = layers.Dropout(dropout * 2)(inputs)

    for i in range(num_layers):
        gru = layers.GRU(
            hidden_dim,
            return_sequences=(i < num_layers - 1),
            dropout=dropout if num_layers > 1 else 0.0,
            recurrent_dropout=0.0,
            reset_after=True,
            implementation=1,
            name=f'gru_{i}'
        )
        if bidirectional:
            x = layers.Bidirectional(gru)(x)
        else:
            x = gru(x)

    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation=None, name='output')(x)

    return models.Model(inputs, outputs)