# numpy_neural_net
A simple neural network (multilayer perceptron) with backpropagation implemented in Python with NumPy



something something a simple 3-layer neural network implemented from scratch in numpy

used (guy's post) and goodfllow's book as inspo

performance shown below (generate plots on the moons data + maybe another classification example)

parameter tuning looked as follows

lessons learned:
* sigmoid functions really do complicate multilayer perceptrons. too low of a learning rate, too many observations, and xyz all made this model highly unstable, and even broke it in some cases. 

* these these models really are incredibly flexible. This simple network was able to approximate every function I threw its way

* neural networks are hard. I have a newfound appreciation for the layers of abstraction that tensorflow, keras, etc. provide between programmer and network
