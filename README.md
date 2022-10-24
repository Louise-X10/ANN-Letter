# ANN-Letter Classification

`hw3.py` reads in data from `letter-recognition.data` and `letter-recognition.name` to perform ANN training and validation. The ANN has 2 hidden layers, and all layers of neutron implement sigmoid activation function. Softmax is applied to output layer for readable output. The default number of epochs is 10,000. 

## Output
The code will produce four separate validation accuracy plots for learning rate mu values `{0.5, 0.1, 0.01, 0.001}`. Each plot has 4 curves for 4 batch sizes `{1, 10, 100, 500}`. 
- `./CMs`: the 16 confusion matrices for each combination of mu and batch size 
- `./CMs_gray`: the gray-scale 2D matrices for each combination of mu and batch size 
