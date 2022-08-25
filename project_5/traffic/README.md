# Traffic

## First attempt - Naive test

### Model 0
For the first attempt I used the same configuration shown in the lecture.
```
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128, activation="relu"),
tf.keras.layers.Dropout(0.5),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```
The resulting model had a 5.7% accuracy on the training set and 5.54% on the testing set. Obviously this is not suitable for any real application.\
After multiple executions of the same model, the best result had an accuracy of 30% on the training set and 38,8% on the testing set but the majority of the results was still around a 5% accuracy.\
A possible reason of this discrepancy could be the choice of the initial weights.


## Second attempt - Drop out tweaks

### Model 1.0
For the second attempt I tried to run `model0` without any drop out of the hidden layer.
```
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128, activation="relu"),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```
The result was a model with 97% accuracy on training set and 92% on testing set which is definitely better than the previous attempt.\
But by removing the drop out, the model could be more "vulnerable" to overfitting.

### Model 1.0_20
To see if the result could be improved, I tried to train the model for 20 epochs.
The resulting model had an accuracy of 97.7% on training set and 92.8% on testing set. Roughly the same as the model trained for 10 epochs.

### Model 1.1
To see if the model had any overfitting problem, I executed the same model with a different dataset proportion (40% training and 60% testing).
The result had an accuracy of 90.0% on training set and 86.6% on testing which, in my opinion, is acceptable.


## Third attempt - Hidden layer tweaks

### Model 2.0
Considered that the neural network used during the lesson was designed to recognise handwriting on greyscale images, while this project uses more complex images, I tried increasing (doubling) the number of nodes in the hidden layer.
```
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(256, activation="relu"),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```
The resulting model had an accuracy of 97.7% on training set and 91.8% on testing set, quite similar to `model1.*`.

### Model 2.1
Now that the neural network has more hidden layer nodes, I tried to put drop out back.
```
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(256, activation="relu"),
tf.keras.layers.Dropout(0.5),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```
The result had an accuracy of 69.7% on training set and 87% on testing set.

### Model 2.2
Suspecting that `model2.1` was dropping out too many nodes, I tried reducing the dropped out proportion to 0.3, obtaining an accuracy of 77% on training set and 86% on testing set.

### Model 2.2_20
The accuracy of `model2.2` was significantly increasing in each epoch, so I tried to train the same model for 20 epochs.
The resulting model had an accuracy of 91.9% on training set and 94.6% on testing set which is significantly better than the model with 10 epochs.


## Activation function
In this section I will analyze the accuracy of the models by changing the activation function.\
According to [this thread](https://datascience.stackexchange.com/questions/10048/what-is-the-best-keras-model-for-multi-class-classification), softmax is a good choice for multiclass classification, so I will not change the output layer's activation function.

### Model Tanh
The resulting model had an accuracy of 5% on both training and testing set.
```
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128, activation="tanh"),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```
A similar result is obtained by doubling the hidden layer's nodes.

### Model SELU
The resulting model had an accuracy of 96% on training set and 92.8% on testing set.\
This is expected since ReLU and SELU have similar form.
```
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128, activation="selu"),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```

### Model ELU
The resulting model had an accuracy of 96% on training set and 91.8% on testing set.\
As with RELU, this is expected since ReLU and ELU have similar form.
```
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128, activation="selu"),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```

### Model Linear
The resulting model had an accuracy of 94% on training set and 89.5% on testing set.\
A linear function could be seen as restriction of ReLU on the domain [0, +inf[ and for this reason the result is still similar to the previous attempts. 
```
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128, activation="linear"),

tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
```