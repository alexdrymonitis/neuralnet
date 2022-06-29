#neuralnet
##An artificial Neural Network framework for Pure Data
[neuralnet] is an artificial neural network Pd external, written in pure C, without any dependencies. It is inspired by the book "Neural Networks from Scratch in Python" by Harrison Kinsley & Daniel Kukieła. It is an attempt to translate the Python code to C with the Pure Data API, to run neural networks within Pd.

[neuralnet] creates densely connected neural networks for classification, regression, and binary logistic regression. There are different activation functions and optimizers you can set, and various other settable parameters. The object's help patch and the examples found in the examples directory should cover all the necessary information.

##Note about the examples
Example 03-mouse_input.pd uses [mousestate] from the Cyclone library, to get the coordinates of the mouse.
Example 04-fahion_mnist.pd uses the [command] external, plus some Python scripts (called via [command]).
Example 05-accelerometer_input.pd uses a mobile app to send accelerometer values via OSC.

All external objects used in the examples can be installed via the deken plugin (Help->Find externals).

Written by Alexandros Drymonitis
