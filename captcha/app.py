from flask import Flask

import numpy as np
from tensorflow import keras


app = Flask(__name__)

print("load MNIST data")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
X = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

# only 0, 1, and 2 to speed things up
ind = (y < 3)
X = X[ind]
y = y[ind]

n = y.shape[0]


@app.get("/captcha")
def print_captcha():
  r = np.random.randint(0, n)
  return {"id": r, "data": X[r,:,:].flatten().tolist()}

@app.post("/captcha/<id>/response/<pred>")
def solve_captcha(id, pred):
  print(y[int(id)])
  print(pred)
  return {"reward": int(y[int(id)] == int(pred))}


if __name__ == '__main__':
  app.run()
