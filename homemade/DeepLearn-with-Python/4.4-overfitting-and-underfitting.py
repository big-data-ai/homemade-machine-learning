import keras

from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1.  # set specific indices of results[i] to 1s
            print(results)
            return results

        # Our vectorized training data
        x_train = vectorize_sequences(train_data)
        # Our vectorized test data
        x_test = vectorize_sequences(test_data)
        # Our vectorized labels
        y_train = np.asarray(train_labels).astype('float32')
        y_test = np.asarray(test_labels).astype('float32')

        from keras import models
        from keras import layers

        original_model = models.Sequential()
        original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        original_model.add(layers.Dense(16, activation='relu'))
        original_model.add(layers.Dense(1, activation='sigmoid'))
        print(original_model.summary())

        original_model.compile(optimizer='rmsprop',
                            loss='binary_crossentropy',
                            metrics=['acc'])

        smaller_model = models.Sequential()
        smaller_model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
        smaller_model.add(layers.Dense(4, activation='relu'))
        smaller_model.add(layers.Dense(1, activation='sigmoid'))
        print(smaller_model.summary())

        smaller_model.compile(optimizer='rmsprop',
                            loss='binary_crossentropy',
                            metrics=['acc'])

        original_hist = original_model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_test, y_test))
        smaller_model_hist = smaller_model.fit(x_train, y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_test, y_test))

        epochs = range(1, 21)
        original_val_loss = original_hist.history['val_loss']
        smaller_model_val_loss = smaller_model_hist.history['val_loss']

        import matplotlib.pyplot as plt

        # b+ is for "blue cross"
        plt.plot(epochs, original_val_loss, 'b+', label='Original model')
        # "bo" is for "blue dot"
        plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
        plt.xlabel('Epochs')
        plt.ylabel('Validation loss')
        plt.legend()

        plt.show()

        bigger_model = models.Sequential()
        bigger_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
        bigger_model.add(layers.Dense(512, activation='relu'))
        bigger_model.add(layers.Dense(1, activation='sigmoid'))

        bigger_model.compile(optimizer='rmsprop',
                            loss='binary_crossentropy',
                            metrics=['acc'])

        bigger_model_hist = bigger_model.fit(x_train, y_train,
                                            epochs=20,
                                            batch_size=512,
                                            validation_data=(x_test, y_test))
        bigger_model_val_loss = bigger_model_hist.history['val_loss']

        plt.plot(epochs, original_val_loss, 'b+', label='Original model')
        plt.plot(epochs, bigger_model_val_loss, 'bo', label='Bigger model')
        plt.xlabel('Epochs')
        plt.ylabel('Validation loss')
        plt.legend()

        plt.show()

        original_train_loss = original_hist.history['loss']
        bigger_model_train_loss = bigger_model_hist.history['loss']

        plt.plot(epochs, original_train_loss, 'b+', label='Original model')
        plt.plot(epochs, bigger_model_train_loss, 'bo', label='Bigger model')
        plt.xlabel('Epochs')
        plt.ylabel('Training loss')
        plt.legend()

        plt.show()

        from keras import regularizers

        l2_model = models.Sequential()
        l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                                activation='relu', input_shape=(10000,)))
        l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                                activation='relu'))
        l2_model.add(layers.Dense(1, activation='sigmoid'))
        l2_model.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['acc'])

        l2_model_hist = l2_model.fit(x_train, y_train,
                                epochs=20,
                                batch_size=512,
                                validation_data=(x_test, y_test))
        l2_model_val_loss = l2_model_hist.history['val_loss']

        plt.plot(epochs, original_val_loss, 'b+', label='Original model')
        plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')
        plt.xlabel('Epochs')
        plt.ylabel('Validation loss')
        plt.legend()

        plt.show()


        # L1 regularization
        regularizers.l1(0.001)

        # L1 and L2 regularization at the same time
        regularizers.l1_l2(l1=0.001, l2=0.001)

        # At training time: we drop out 50% of the units in the output
        layer_output *= np.randint(0, high=2, size=layer_output.shape)
        # At test time:
        layer_output *= 0.5

        # At training time:
        layer_output *= np.randint(0, high=2, size=layer_output.shape)
        # Note that we are scaling *up* rather scaling *down* in this case
        layer_output /= 0.5
        model.add(layers.Dropout(0.5))

        dpt_model = models.Sequential()
        dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        dpt_model.add(layers.Dropout(0.5))
        dpt_model.add(layers.Dense(16, activation='relu'))
        dpt_model.add(layers.Dropout(0.5))
        dpt_model.add(layers.Dense(1, activation='sigmoid'))

        dpt_model.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['acc'])

        dpt_model_hist = dpt_model.fit(x_train, y_train,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(x_test, y_test))

        dpt_model_val_loss = dpt_model_hist.history['val_loss']

        plt.plot(epochs, original_val_loss, 'b+', label='Original model')
        plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
        plt.xlabel('Epochs')
        plt.ylabel('Validation loss')
        plt.legend()

plt.show()
