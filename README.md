# neuralnet
Version 0.2
## An Artificial Neural Network framework for Pure Data
[neuralnet] is an artificial neural network Pd external, written in pure C, without any dependencies. It is inspired by the book "Neural Networks from Scratch in Python" by Harrison Kinsley & Daniel Kukieła. It is an attempt to translate the Python code to C with the Pure Data API, to run neural networks within Pd.

[neuralnet] creates densely connected neural networks for classification, regression, and binary logistic regression. There are different activation functions and optimizers you can set, and various other settable parameters. The object's help patch and the examples found in the examples directory should cover all the necessary information.

## Note about Make
This repository uses the pd-lib-builder Makefile system. You can get it from [here](https://github.com/pure-data/pd-lib-builder). The directory of the Makefile should be in the same directory of the neuralnet directory. For example, run:
```
cd ../
git clone https://github.com/pure-data/pd-lib-builder.git
cd -
```

## Note about the examples
Example 03-mouse_input.pd uses [mousestate] from the Cyclone library, to get the coordinates of the mouse.
Example 04-fahion_mnist.pd uses the [command] external, plus some Python scripts (called via [command]).
Example 05-accelerometer_input.pd uses a mobile app to send accelerometer values via OSC.

All external objects used in the examples can be installed via the deken plugin (Help->Find externals).

If you mention this object in an academic paper/chapter/article, please include it in your bibliography with the following citation:

```
@article{Drymonitis2023neuralnet,
	author = {Drymonitis, Alexandros},
	journal = {AIMC 2023},
	year = {2023},
	month = {aug 29},
	note = {https://aimc2023.pubpub.org/pub/3j3fx7y1},
	publisher = {},
	title = {[neuralnet]: A {Pure} {Data} {External} for the {Creation} of {Neural} {Networks} {Written} in {Pure} {C}},
}
```

Log:<br />
-Fixes crash on Windows with the set_activation_function() message that took A_FLOAT and A_SYMBOL as arguments, which apparently in Windows is not possible.

Written by Alexandros Drymonitis
