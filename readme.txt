The program is running under python. Package Numpy and pickle should be installed.

Also you should include the file ¡®MNIST_test.pkl¡¯, ¡®MNIST_train.pkl¡¯, ¡®MNIST_valid.pkl¡¯ and ¡®validate_data.pkl¡¯ under the same folder. 

There are two files you can run, one is tester.py and another is neuralnet.py. tester.py is for part b and neuralnet.py is for part c to f.

neuralnet.py will print a number after done. It is the accuracy for the test set. To run it, you can type ¡®python neuralnet.py¡¯ to run in the terminal, and then the final result will be printed.



*******
Following is for tester.py.

'tester.py' is written for checking whether the code in neuralnet.py is correct and safe to use when calculating the gradients of tanh, sigmoid and ReLU activation function. This python file should be placed under the same directory as neuralnet.py as it imports neuralnet.py.

When executing this file, ther are 4 variables that can be alternated to test the correctness of neuralnet.py. The first one is called 'op' whose value can be chosen from 0, 1, 2, 3. '0' means we choose an input to hidden weight to test. '1' means we choose a hidden to output weight to test. '2' means we choose a hidden bias weight to test and '3' means we choose an output bias weight to test. The programme tests only one weight each time it runs.

The 'epsilon' variable sets the value of a small constant added or subtracted from the chosen weight. In our test we set it as 0.01. The gradient calculated by numerical approximation and back propagation should be within the scope of O(epsilon^2) so that the neuralnet.py is correct.
The value of epsilon can be set as any small constant.

The config['layer_specs'] and config['activation'] variables determine the network topology and activation function used. They can be alternated at will.

At last, the test set can be set in line 27(' X_test, y_test = neuralnet.load_data('MNIST_test.pkl') # test data '). Just make sure the test set is available to neuralnet.py to load. 