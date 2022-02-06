from keras.models import Sequential
from keras.layers import Dense

class NumberBandit:
    def __init__(self):
        self.net = self.make_net()

    def make_net(self):
        Bandit = Sequential()
        Bandit.add(Dense(128, activation='sigmoid'))
        Bandit.add(Dense(1, activation='sigmoid'))
        Bandit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return Bandit

    def train_net(self, input_batch, target_batch):
        self.net.fit(input_batch, target_batch, batch_size=len(input_batch), epochs=10, verbose=False)

    def make_prediction(self, input):
        return self.net.predict(input, batch_size=len(input))
