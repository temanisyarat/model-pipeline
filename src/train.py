import tensorflow as tf


def train_tf_model(train_ds, val_ds, CONFIG, num_classes, input_dim, steps_per_epoch=None, validation_steps=None):
    """Training loop for TensorFlow model."""
    from .model import build_mobile_sign_gru

    model = build_mobile_sign_gru(
        input_dim,
        num_classes,
        CONFIG['max_len'],
        CONFIG['hidden_dim'],
        CONFIG['num_layers'],
        CONFIG['dropout'],
        CONFIG['bidirectional']
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=CONFIG['patience'],
            restore_best_weights=True,
            monitor='val_loss'
        )
    ]

    for x, y in train_ds.take(1):
        print(x.shape, y.shape)

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=CONFIG['epochs'],
        callbacks=callbacks
    )

    return model, history