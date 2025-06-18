
Install librosa, glob, PyTorch, torchaudio, pandas, numpy.

Invocate "test" function to run the code after changing the path varaibles in test file.

Evaluate and test functions have been used, passing data seperately for test(not in batch).

First model is included by default, if wanted can be replaced by second model right inside the test.py file. If the architecture is included the model.pt file can be changed and the model could be run. 
Thus, just change the architecture and the model_CNN1.pt file to model_CNN2.pt run the other model.

The pre- defined variables have been replaced by us, use a .csv file path in one and the test folder path in another variable. 

Also, the second argument in torch.load can be removed if gpu is being used to train the model.

The model architectures have been included in test file so as to make the model_CNN.pt files usable.



