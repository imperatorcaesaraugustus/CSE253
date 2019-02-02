import numpy as np
import pickle



config = {}
config['layer_specs'] = [784, 48, 36, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'ReLU' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.5 # Learning rate of gradient descent algorithm

class normailize_image():
  def __init__(self):
    self.Max = None
    self.Min = None
    self.Middle = None
  def normal_init(self,X_need):
    self.Max = np.max(X_need, axis=0)
    self.Min = np.min(X_need, axis=0)
    self.Middle = (np.max(X_need, axis=0)+np.min(X_need, axis=0))/2
    return 2*(X_need-self.Middle)/np.where(self.Max-self.Min,self.Max-self.Min,1)
  def normal(self,X_need):
    return 2*(X_need-self.Middle)/np.where(self.Max-self.Min,self.Max-self.Min,1)

norm = normailize_image()

def softmax(x):
  # row: DATA; column: CLASS
  output = np.exp(x)
  desum = np.array(output.sum(axis=1))
  #print(desum.shape,output.shape)
  output = ((output.T)/desum).T
  return output


def load_data(fname):
  file = open(fname,'rb')
  f = pickle.load(file)
  images = f[:,:-1]
  labels = f[:,-1]
  return images, labels


class Activation:
  def __init__(self, activation_type="sigmoid"):
    self.activation_type = activation_type
    self.x = None  # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)

    elif self.activation_type == "tanh":
      return self.tanh(a)

    elif self.activation_type == "ReLU":
      return self.ReLU(a)

  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()

    elif self.activation_type == "tanh":
      grad = self.grad_tanh()

    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()

    return np.multiply(grad, delta)

  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    # row: data; column: 1
    self.x = x
    output = np.ones(x.shape) / (np.ones(x.shape) + np.exp(-x))
    return output

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    sig = self.sigmoid(self.x)
    grad = sig * (np.ones(self.x.shape) - sig)
    return grad

  def tanh(self, x):
    self.x = x
    output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return output

  def grad_tanh(self):
    tanhs = self.tanh(self.x)
    grad = np.ones(tanhs.shape) - tanhs * tanhs
    return grad

  def ReLU(self, x):
    self.x = x
    output = np.maximum(0, x)
    return output

  def grad_ReLU(self):
    grad = np.where(self.x > 0, 1, 0)
    return grad


class Layer():
  def __init__(self, in_units, out_units):
    # np.random.seed(42)
    self.w = np.random.randn(in_units, out_units) *0.01  # Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this

    self.v_dw = np.zeros(self.w.shape)
    self.v_db = np.zeros(self.b.shape)

    self.best_w = None
    self.best_b = None

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    self.a = self.x.dot(self.w) + self.b
    return self.a

  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    lam = config['L2_penalty']
    self.d_w = np.dot(self.x.T, delta) - lam * self.w / len(self.x)
    self.d_b = np.dot(np.ones((1, len(self.x))).astype(np.float32), delta)
    self.d_x = np.dot(delta, self.w.T)

    return self.d_x

  def update_w_b(self, alpha, gamma):
    self.v_dw = self.d_w * (1 - gamma) + gamma * self.v_dw
    self.v_db = self.d_b * (1 - gamma) + gamma * self.v_db
    self.w = self.w + self.v_dw * alpha
    self.b = self.b + self.v_db * alpha

  def regularization(self):
    output = np.sum(np.reshape(np.square(self.w), (self.w.size,))) / len(self.x)
    # output += np.sum(np.reshape(np.square(self.b),(self.b.size,)))
    return output

  def this_is_best(self):
    self.best_w = np.copy(self.w)
    self.best_b = np.copy(self.b)

  def change_to_best(self):
    self.w = np.copy(self.best_w)
    self.b = np.copy(self.best_b)


class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))

  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x
    Outy = x.copy()
    NumLayers = len(config['layer_specs']) - 2
    for i in range(NumLayers):
      TmpLayer = self.layers[2 * i]
      TmpAct = self.layers[2 * i + 1]
      Outy = TmpLayer.forward_pass(Outy)
      Outy = TmpAct.forward_pass(Outy)
    Outy = self.layers[-1].forward_pass(Outy)
    self.y = softmax(Outy)
    loss = 0
    if not targets is None:
      self.targets = targets
      loss = self.loss_func(self.y, targets)
    return loss , self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    self.targets = targets
    tkyk = np.multiply(np.log(logits), targets)
    ntimesk = tkyk.shape[0] * tkyk.shape[1]
    output = -np.sum(np.reshape(tkyk, (tkyk.size,))) / ntimesk
    return output

  def backward_pass(self):
    '''
    implement the backward pass for the whole network.
    hint - use previously built functions.
    '''
    delta = (self.targets - self.y) / len(self.y)
    delta = self.layers[-1].backward_pass(delta)
    NumLayers = len(config['layer_specs']) - 2
    for i in range(NumLayers)[::-1]:
      TmpLayer = self.layers[2 * i]
      TmpAct = self.layers[2 * i + 1]
      delta = TmpAct.backward_pass(delta)
      delta = TmpLayer.backward_pass(delta)

  def update_w_b(self, alpha, gamma):
    NumLayers = len(config['layer_specs']) - 1
    for i in range(NumLayers):
      TmpLayer = self.layers[2 * i]
      TmpLayer.update_w_b(alpha, gamma)

  def this_is_best(self):
    NumLayers = len(config['layer_specs']) - 1
    for i in range(NumLayers):
      TmpLayer = self.layers[2 * i]
      TmpLayer.this_is_best()

  def change_to_best(self):
    NumLayers = len(config['layer_specs']) - 1
    for i in range(NumLayers):
      TmpLayer = self.layers[2 * i]
      TmpLayer.change_to_best()


def trainer(nnet, X_train, y_train, X_valid, y_valid, config):
  X_train = norm.normal_init(X_train)
  X_valid = norm.normal(X_valid)

  alpha = config['learning_rate']
  gamma = config['momentum_gamma'] if config['momentum'] else 0
  batch_size = config['batch_size']
  batch_num = int((len(X_train) - 1) / batch_size) + 1
  training_loss = []
  validation_loss = []
  training_accu = []
  validation_accu = []
  y_valid_onehot = [[1 if i == x else 0 for i in range(10)] for x in y_valid]
  last_loss = 1000000
  best_loss = 1000000
  increasing_num = 0
  for _ in range(config['epochs']):
    data_train = np.c_[X_train, y_train]
    np.random.shuffle(data_train)
    min_batches = [data_train[i * batch_size:(i + 1) * batch_size] for i in range(batch_num)]
    accu = 0
    for small_batch in min_batches:
      X_train_minbatch = small_batch[:, :-1]
      y_train_minbatch = [[1 if i == x else 0 for i in range(10)] for x in small_batch[:, -1]]
      loss, y_value = nnet.forward_pass(X_train_minbatch, targets=y_train_minbatch)
      nnet.backward_pass()
      nnet.update_w_b(alpha, gamma)
      y_value = y_value.argmax(axis=1)
      accu += sum([y_train_minbatch[i][y_value[i]] == 1 for i in range(len(y_train_minbatch))])
    training_accu.append(accu / len(y_train))
    training_loss.append(loss)
    loss, y_value = nnet.forward_pass(X_valid, targets=y_valid_onehot)
    validation_loss.append(loss)
    y_value = y_value.argmax(axis=1)
    accu = sum([y_value[i] == y_valid[i] for i in range(len(y_valid))])
    validation_accu.append(accu / len(y_valid))

    if validation_loss[-1] > last_loss:
      increasing_num += 1
      if increasing_num == config['early_stop_epoch']:
        break
    else:
      increasing_num = 0
      if validation_loss[-1] < best_loss:
        nnet.this_is_best()
        best_loss = validation_loss[-1]
    last_loss = loss
  nnet.change_to_best()



def test(nnet, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  X_test = norm.normal(X_test)
  y_test_onehot = [[1 if i == x else 0 for i in range(10)] for x in y_test]
  _, y_value = nnet.forward_pass(X_test, targets=y_test_onehot)
  y_value = y_value.argmax(axis=1)
  accuracy = sum([y_value[i] == y_test[i] for i in range(len(y_test))]) / len(y_test)
  print("Test Accuracy is", accuracy)
  return accuracy

if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'

  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)

