import neuralnet
import numpy as np
import pickle

def loss_func(y, t):
    output = 0
    for i in range(len(t)):
        output += (y[0][i] - t[i])*(y[0][i] - t[i])/2.0
    return output    

def main():
    # make_pickle()
    benchmark_data = pickle.load(open('validate_data.pkl', 'rb'), encoding='latin1')
    config = {}
    config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
    config['activation'] = 'tanh' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
    input_size = config['layer_specs'][0]
    hidden_size = config['layer_specs'][1]
    output_size = config['layer_specs'][2]

    np.random.seed(42)
    w1 = np.random.randn(input_size, hidden_size)  # Weight matrix 1
    b1 = np.random.randn(1,1).dot(np.ones((1, hidden_size)).astype(np.float32))  # Bias vector 1
    w2 = np.random.randn(hidden_size, output_size)  # Weight matrix 2
    b2 = np.random.randn(1,1).dot(np.ones((1, output_size)).astype(np.float32))   # Bias vector 2

    X_test, y_test = neuralnet.load_data('MNIST_test.pkl') # test data
    y_test_onehot = [[1 if i == y else 0 for i in range(10)] for y in y_test]
    act = neuralnet.Activation('tanh')
    epsilon = 0.01  # epsilon value
    x_index, y_index = 0, 3
    op = 0  # 0:input to hidden weight; 1:hidden to output weight; 2:hidden bias weight; 3:hidden bias weight

    ops = [[epsilon,0,0,0],[0,epsilon,0,0],[0,0,epsilon,0],[0,0,0,epsilon]]

    w1[x_index][y_index] += ops[op][0]  # increase the weight by epsilon
    w2[x_index][y_index] += ops[op][1]
    for b in b1: b += ops[op][2]
    for b in b2: b += ops[op][3]
    input_h = np.dot(X_test[0], w1) + b1
    output_h = act.sigmoid(input_h)
    input_o = np.dot(output_h, w2) + b2
    output_o = act.sigmoid(input_o)
    loss1 = loss_func(output_o, y_test_onehot[0])

    w1[x_index][y_index] -= 2*ops[op][0]  # decrease the weight by 2 epsilon
    w2[x_index][y_index] -= 2*ops[op][1]
    for b in b1: b -= 2*ops[op][2]
    for b in b2: b -= 2*ops[op][3]
    input_h = np.dot(X_test[0], w1) + b1
    output_h = act.sigmoid(input_h)
    input_o = np.dot(output_h, w2) + b2
    output_o = act.sigmoid(input_o)
    loss2 = loss_func(output_o, y_test_onehot[0])
    diff1 = (loss1 - loss2)/(2*epsilon)

    w1[x_index][y_index] += ops[op][0]  # recover the weight
    w2[x_index][y_index] += ops[op][1]
    for b in b1: b += ops[op][2]
    for b in b2: b += ops[op][3]
    input_h = np.dot(X_test[0], w1) + b1
    output_h = act.sigmoid(input_h)
    input_o = np.dot(output_h, w2) + b2
    output_o = act.sigmoid(input_o)
    

    # choose hidden weight
    if op == 0:
        ans = 0
        for l in range(10):
            ans += (w2[y_index][l]*(output_o[0][l] - y_test_onehot[0][l])*output_o[0][l]*(1 - output_o[0][l]))
        ans *= (output_h[0][y_index]*(1 - output_h[0][y_index])*X_test[0][x_index])
        diff2 = ans
    # choose output weight
    elif op == 1:
        diff2 = (output_o[0][y_index] - y_test_onehot[0][y_index])*output_o[0][y_index]*(1 - output_o[0][y_index])*output_h[0][x_index]
    # choose input bias weight
    elif op == 2:
        diff2 = 0
        for i in range(50):
            ans = 0
            for l in range(10):
                ans += (w2[i][l]*(output_o[0][l] - y_test_onehot[0][l])*output_o[0][l]*(1 - output_o[0][l]))
            ans *= output_h[0][i]*(1 - output_h[0][i])
            diff2 += ans 
    # choose hidden bias weight
    else:
        diff2 = 0
        for i in range(10):
            diff2 += (output_o[0][i] - y_test_onehot[0][i])*output_o[0][i]*(1 - output_o[0][i])
        
    print(abs(diff1), abs(diff2))
    print("difference :", abs(diff1 - diff2))
  
  
if __name__ == '__main__':
    main()


