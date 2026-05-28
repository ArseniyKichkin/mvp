import mlflow.pyfunc
import pandas as pd
from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
import os
import logging
import csv
from datetime import datetime

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')

all_windows_list = []

class Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load custom artifacts if necessary
        pass

    def predict(self, context, model_input):
        save_dir = "/app/inference_preds"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, "predictions.csv")
        n_current_anomalies = len(self.vae_model.inference_scores)
        prediction = self.lstm_model.predict(model_input[:,:-1,:])[0].item()
        error = np.abs(prediction - model_input[0, -1, 0])
        self.inference_window.append(error)
        with open(filename, 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['timestamp', 'prediction', 'error', 'window_length'])

            # Записываем данные
            writer.writerow([
                datetime.now().isoformat(),
                prediction,
                error,
                len(self.inference_window)
            ])
        all_windows_list.append(prediction)
        if len(self.inference_window) == 50:
            self.inference_window = pd.Series(self.inference_window).ewm(span=int(70 * 30 * 0.05)).mean().values.flatten()
            self.vae_model.test_model(self.inference_window, is_inference=True)
            self.inference_window = []
        if len(self.vae_model.inference_scores) > n_current_anomalies:
            return 1, self.vae_model.inference_scores
        return 0, self.vae_model.inference_scores

    def __init__(self, config, run_id, channel):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test model_data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.y_hat_train = np.array([])
        self.lstm_model = None
        self.vae_model = None
        self.inference_window = []

        if not self.config.train:
            try:
                self.load()
            except FileNotFoundError:
                path = os.path.join('data', self.config.use_id, 'models',
                                    self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(path))
                self.train_new(channel)
                self.save()
        else:
            self.train_new(channel)
            self.save()

    def load(self):
        """
        Load model for channel.
        """

        logger.info('Loading pre-trained model')
        self.lstm_model = load_model(os.path.join('data', self.config.use_id,
                                             'models', self.chan_id + '.h5'))

    def train_new(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test model_data
                for X,y for a single channel
        """

        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=self.config.patience,
                                        min_delta=self.config.min_delta,
                                        verbose=0)]

        self.lstm_model = Sequential()

        self.lstm_model.add(LSTM(
            self.config.layers[0],
            input_shape=(None, channel.X_train.shape[2]),
            return_sequences=True))
        self.lstm_model.add(Dropout(self.config.dropout))

        self.lstm_model.add(LSTM(
            self.config.layers[1],
            return_sequences=False))
        self.lstm_model.add(Dropout(self.config.dropout))

        self.lstm_model.add(Dense(
            self.config.n_predictions))
        self.lstm_model.add(Activation('linear'))

        self.lstm_model.compile(loss=self.config.loss_metric,
                           optimizer=self.config.optimizer)

        self.lstm_model.fit(channel.X_train,
                       channel.y_train,
                       batch_size=self.config.lstm_batch_size,
                       epochs=self.config.epochs,
                       validation_split=self.config.validation_split,
                       callbacks=cbs,
                       verbose=True)

    def save(self):
        """
        Save trained model.
        """

        self.lstm_model.save(os.path.join('data', self.run_id, 'models',
                                     '{}.h5'.format(self.chan_id)))

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def aggregate_predictions_during_train(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat_train = np.append(self.y_hat_train, agg_y_hat_batch)

    def batch_predict(self, channel):
        """
        Used trained LSTM model to predict test model_dataarriving in batches.

        Args:
            channel (obj): Channel class object containing train/test model_data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                          / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_test.shape[0]))

        # simulate model_data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                idx = channel.y_test.shape[0]
            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.lstm_model.predict(X_test_batch)
            self.aggregate_predictions(y_hat_batch)

        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        channel.y_hat = self.y_hat

        np.save(os.path.join('data', self.run_id, 'y_hat', '{}.npy'
                             .format(self.chan_id)), self.y_hat)

        return channel

    def batch_predict_during_train(self, channel):
        """
        Used trained LSTM model to predict train model_data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test model_data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = int((channel.y_train.shape[0] - self.config.l_s)
                          / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_train.shape[0]))

        # simulate model_data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                idx = channel.y_train.shape[0]

            X_train_batch = channel.X_train[prior_idx:idx]
            y_hat_batch = self.lstm_model.predict(X_train_batch)
            self.aggregate_predictions_during_train(y_hat_batch)

        self.y_hat_train = np.reshape(self.y_hat_train, (self.y_hat_train.size,))

        channel.y_hat_train = self.y_hat_train

        np.save(os.path.join('data', self.run_id, 'y_hat_train', '{}.npy'
                             .format(self.chan_id)), self.y_hat_train)

        return channel