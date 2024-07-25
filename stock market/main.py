# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from termcolor import cprint


# RNN

# apple_share_price.csv


class RRN:
    def __init__(self, filename):
        self.dataset = pd.read_csv(filename, usecols=[1, 2, 3, 4])
        self.dataset = self.dataset.reindex(index=self.dataset.index[::-1])
        cprint("Dataset imported successfully", "green")

    def index(self):
        self.obs = np.arange(1, len(self.dataset) + 1, 1)
        cprint("Arranged", "green")

    def plot(self):

        print(self.dataset.columns)

        plt.plot(self.obs, self.dataset["Close"])
        plt.title("Open/Close price of stocks")
        plt.xlabel("Time (latest-> oldest)")
        plt.ylabel("Price of stocks")
        plt.show()
        cprint("Plot", "green")

    def indicators(self):
        self.OHLC_avg = self.dataset.mean(axis=1)
        self.HLC_avg = self.dataset[["High", "Low", "Close"]].mean(axis=1)
        self.close_val = self.dataset[["Close"]]
        cprint("Indicators", "green")

    def prep(self):
        self.OHLC_avg = np.reshape(self.OHLC_avg.values, (len(self.OHLC_avg), 1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.OHLC_avg = scaler.fit_transform(self.OHLC_avg)
        cprint("Preparation", "green")

    def split(self):

        self.train_OHLC = int(len(self.OHLC_avg) * 0.75)
        self.test_OHLC = len(self.OHLC_avg) - self.train_OHLC
        self.train_OHLC, self.test_OHLC = (
            self.OHLC_avg[0 : self.train_OHLC, :],
            self.OHLC_avg[self.train_OHLC : len(self.OHLC_avg), :],
        )
        cprint("Train Test", "green")

    def reshape(self):
        self.trainX = self.train_OHLC[:-1]
        self.trainY = self.train_OHLC[1:]
        self.testX = self.test_OHLC[:-1]
        self.testY = self.test_OHLC[1:]
        cprint("Set up training + test vars", "green")

        self.trainX = np.reshape(
            self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1])
        )
        self.testX = np.reshape(
            self.testX, (self.testX.shape[0], 1, self.testX.shape[1])
        )
        self.step_size = 1
        cprint("Reshaping", "green")

    def model_train(self):
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(1, self.step_size), return_sequences=True))
        self.model.add(LSTM(16))
        self.model.add(Dense(1))
        self.model.add(Activation("linear"))
        cprint("LSTM", "green")

        self.__compile()
        self.model.fit(self.trainX, self.trainY, epochs=5, batch_size=1, verbose=2)
        cprint("compile", "green")

    def model_open(self, filename):
        self.model = keras.models.load_model(filename)

    def __compile(self):
        self.model.compile(loss="mean_squared_error", optimizer="adagrad")

    def predict(self):
        trainPredict = self.model.predict(self.trainX)
        testPredict = self.model.predict(self.testX)

        trainScore = math.sqrt(mean_squared_error(self.trainY, trainPredict))
        testScore = math.sqrt(mean_squared_error(self.testY, testPredict))

        cprint("Train RMSE: %.2f" % trainScore, "red")
        cprint("Test RMSE: %.2f" % testScore, "red")

    def test(self):
        results = self.model.evaluate(self.testX, self.testY, verbose=2)

        cprint(f"test loss {results}", "red")


def main() -> None:
    filename = "apple_share_price.csv"
    rrn = RRN(filename)
    rrn.index()
    rrn.indicators()
    rrn.prep()
    rrn.split()
    rrn.reshape()
    rrn.model_train()

    rrn.test()

    rrn.predict()


if __name__ == "__main__":
    main()
