# ANN-Letter Classification

`hw3.py` reads in data from `letter-recognition.data` and `letter-recognition.data` to perform ANN training and validation. The default number of epochs is 10,000. It will produce four separate validation accuracy plots for learning rate mu values `{0.5, 0.1, 0.01, 0.001}`. Each plot has 4 curves for 4 batch sizes `{1, 10, 100, 500}`. The 16 confusion matrices for each combition of mu and batch size are saved in the directory `./CMs`. The gray-scale 2D matrices are saved in the directory `./CMs_gray`.
