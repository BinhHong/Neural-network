# Neural-network
3 basic steps:
 - chose model
 - define loss function
 - minimize the cost function
 
the result of a layer when fed with values is a tensor and we can turn a ensor into a nupy array by `a.numpy()`
# 2. Preprocessing
## a. Normalization
using 
- `a = tf.keras.layers.Normalization(axis=-1)`
- `a.adapt(X)` with `X` is the training set, to learn mean and variance
- `X_normalized = a(X)` 

# 3. Some details
in `Sequential([...])` we can put in `tf.keras.Input(shape=(a,))` to indicate shape of inputs, also helps instantiate weights

# 4. Vectorization
