Echo-state network architecture for word sense disambiguation.

Use the scripts as follows.

*** For training a model ***
command: python biESN_train.py @parameters_train.txt
the save_path parameter should lead to a folder where the model is to be stored

*** For evaluating a model ***
command: python biESN_test.py @parameters_test.txt > results.txt
the save_path parameter should lead to the pickled file containing the trained model