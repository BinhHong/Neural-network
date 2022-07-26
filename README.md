# Neural-network

# 1. Some conventions
- in `Sequential([...])` we can put in `tf.keras.Input(shape=(a,))` to indicate shape of inputs, also helps instantiate weights
- input for each layer must be 2d




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



# 4. Vectorization


![](pic1.png?raw=true)

Suppose training set is A of the shape $m.n$ matrix, each training example $X= (x_1,...,x_n)$ is a row. 
At each layer, each unit returns a real value by the activation function
$$f(\sum^n_1 w_ix_i + b)$$

- Since A is of $m.n$, the output $a^{[1]}$ of layer1 is obtained by matrix W, where W must be of the form $n.p$, where $p$ is the number of units (e.g n=400, p=25). This means that at each unit, the parameters $w_1,...,w_n$ in activation functions are written as columns in W. 
- B is a collection of biases, so length of B is equal to the number of unit (e.g lenB = 25) and we have
$$a^{[1]} = f(X.W + B)$$ 
- note that B can be 1d (p,) or 2d (p,1)

For summary, W of the size $in.out$ where $in$ and $out$ are the lengths of input and ouput.

# 5. Backward propagation
is used to compute partial derivatives in gradient descent.
# 6. Activation functions
- linear $\sim$ "no activation function"
- sigmoid
- rectified linear unit ReLU
- softmax

Softmax activation function is not a good naming, since at each neutron, the function used is a multivariate function, not univariate, and the functions used at neutrons are totally different. We should use the name `Softmax layer`. 

# 7. Numerical accurate implementation of softmax

- suppose we want to do a multiclass classification (N classes), then we use softmax activation at the final layer with $N$ neutrons. At each neutron, the function use is 
$$a_i=\frac{e^{z_i}}{\sum e^{z_j} }$$
- generally
$$(a_1,...,a_N)=g(z_1,...,z_N)$$
where $g(z_1,...,z_N)= (\frac{e^{z_1}}{\sum e^{z_j}},..., \frac{e^{z_N}}{\sum e^{z_j}})$

Basically, use intermediate variables and then substitute them causes more error. Avoid by using `activation='linear'` instead of `activation='softmax'` and `model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))` instead of `model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))`.

Note that in the Predict section, the old one is `model(X_test)`, which gives values of `a_1, ..., a_N` now gives values of `z_1, ..., z_N`, because of `activation='linear'`. To compute the probabilities, we write 

`logits = model(X_test)`

`f_x = tf.nn.softmax(logits)`

For logistic regression, the last line becomes `f_x = tf.nn.sigmoid(logits)`

# 8. Multi-label classification

consists of several binary classifications. Therefore in the output layer, we use sigmoid activation. Note that the probabilities in this case are separated, their sum is not equal to 1.

# 9. SparseCategoricalCrossentropy vs CategoricalCrossentropy

- SparseCategoricalCrossentropy: use when we expect outputs to be integers corresponding to the indices.
- CategoricalCrossentropy: Expects the target value of an example to be one-hot encoded where the value at the target index is 1 while the other N-1 entries are zero. An example with 10 potential target values, where the target is 2 would be [0,0,1,0,0,0,0,0,0,0].

# 10. Best practices

## a. Basic things
- get more training examples
- smaller set of features
- get additional features
- add polynomial features
- regularized: with $\lambda$ --> increase or decrease $\lambda$
## b. Diagnostic
common mistake: only training and test set. Then new parameters are (unintentionally) added based on test set. For example: linear model --> add $x^2, x^3, x^4,..., x^d,...$ and check test errors to find the smallest one. This is not right since it depends on test set. To avoid this we use cross-validation set (validation set/development set/ dev set) and find smallest validation errors. $d$ found only depends on validation test and not on test set, hence gives a fairly correct one.

High bias (training error $J_t$ is high) vs high variance (validation error $J_v$ >> training error $J_t$):
- $d$ = polynomial degree, runs from 1 to very large, then training error runs from high to low, appro 0. However, validation error runs from high, low at somewhere in middle then high again.
- similar fact happens for regularization parameter $\lambda$, but somewhat like the mirror image of the case of polynomial degree.
- learning curve: graph of $J_t$ and $J_v$ w.r.t number of training examples. We need to define bias/variance depends on a baseline level, that is level set as the benchmark, like human level performance...
- for high bias problem --> adding more training examples doesn't help while for high variance, it might helps.

## c. Debugging a learning algorithm
- high bias: 
  - getting more features
  - adding polynomial features
  - decreasing $\lambda$
- high variance:
  - getting more examples
  - getting smaller set of features
  - increasing $\lambda$
  
## d. Neural networks and regularization
- on a quite small training set, if high bias --> make larger network until high bias is low (at the cost of speed).
- now it might gets high variance --> get more training example until it is good.

Regularization can be applied to the cost function with all the parameters used in all layers, adding 

`tensorflow.keras.regularizers.l2(...)`



# 11. Interative loop of ML development

choose architecture --> train model --> diagnostics (bias/variance and then error analysis, where we look at a sample of where the work is doing wrong and find which are the most common mistakes, then collect more data on these to examine, add, improve..) --> repeat.
# 12. Precision, recall and F1 score
- recall $\sim$ TP rate (for concrete case, not for statistics case): of all patients actually having disease, what fraction is correctly classified?
- precision $\sim$ positive predictive number: of all patiens classied as positive, what fraction actually has disease?
- F1 score = harmonic mean
