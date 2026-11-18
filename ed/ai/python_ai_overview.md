# Search
Used in traffic mapping, creating recipie based on ingredients, routing (supply chain and logstics).  Try to come up with a solution which meets the goal and has a minimal path cost function.
[Overview](0-Search/0%20-%20Search.pdf)

**Depth-First Search** and **Breadth-First Search**
Example building a [maze](0-Search/demo/maze.py)

**Greedy Best Search**
Manhatten distance
Get block distance from goal
When traveling take the number that is the lowest (i.e. closest to the goal)
*A\* Search*: Track number of actions plus the "price" of the square and go with the choice that is cheapest (even as continue down a path)

**Adversarial Search**
When have an opposing force (e.g. a competitor).  e.g. if lower prices what will competitors do and what will be consequences.
Uses *minimax*
*Alpha-Beta Pruning* used to help determin path and lower cost.  See if level is better than originator
**Depth-Limited Minimax** Limit depth
Example: [tic-tac-toe](ai/0 - Search/projects/Tic-Tac-Toe.pdf)

# Knowledge
Try to draw new conclusions based on internal representations of knowledge.
[overview](1-knowledge/1%20-%20Knowledge.pdf)

**Propositional Logic**
Get into CNF (conjuctal normal form)
1. Eliminate biconditionals
2. Eliminate implications
3. Move *not* inwards using De Morgan's Law
4. Use distributive law to  distribute  wherever possible

**First Order Logic**

**Universal Quantification**

# Probabilities
[Overview](2-Uncertainty/2%20-%20Uncertainty.pdf)
**Baynsian Networks**

**Markov Chain**

**Hidden Markov Model**

# Optimization
[Overview](3-Optimization/3%20-%20Optimization.pdf)
e.g. where should I build a new location

**Hill Climbing**
Do neighbors get me into a better spot (heading towards a global maximum or global minimum, depending on objective)

**Linear Programming**
e.g. find best route for a salesman, or how to run 2 machines at different cost and productivity output
Minimize cost function
notebook starting page: 68

# Machine Learning
[Overview](4-Learning/4%20-%20Learning.pdf)

**Supervised Learning**
Machine learn a function to map inputs to outputs
- Classifications: Machine to produce a model to produce labels for data
- - Nearest-neighbor (look at nearest neighbor)
- - k-nearest-neighbor (smooth out, look at net of neighbors) (create hyperplane, pereceptron)
- - Support Vector Machine (SVM).  Create hyperplane to max margin separators
- Regression: Learning teask of learning a function mapping an input point to a continuious value
* Loss Fuction: Express how good/bad function is based on predicted to actual
* * 0-1 loss (1=true, 0 = false)
* * L1 loss (absolute value of actual - predicted)
* * L2 loss (actual - predicted and square the answer).  Really penalizes if further off
* Overfitting
* * Regularization
* Underfitting

**Reinforcement Learning**
set or rewards or punishments to learn what actions to take in the future

**Unsupervised Learning**
Cluster applications.  E.g. genetic research, image segmentation, market research, social network analysis

- *k*-means clustering
Markov Decision Process
Q-learning


# Neural Networks
[Overview](5-Neural_Networks/5%20-%20Neural%20Networks.pdf)
[Tensorflow](playground.tensorflow.org)

* Based on idea of how people thougth neurons worked
**Activation Functions**
* Step Function
* Logistic Sigmoid
* rectified linear unit (ReLU)
**Minimize Loss**
* Gradient Descent
* Stochastic Gradient Descent
* Mini-Batch Gradient Descent

*weights* attach neurons
- Example neuron x1 and x2 = g(w0 + w1x1 + w2x2)
- - e.g. "or" would be w = -1
- - e.g. "and" would be w = -2

**Backpropagation**
Algorithm for training neural networks with hidden layers
• Start with a random choice of weights.
• Repeat:
•   Calculate error for output layer.
•   For each layer, starting with output layer and moving inwards towards earliest hidden layer:
• Propagate error back one layer.
• Update weights

## deep neural networks
neural network with multiple hidden layers

**Overfitting**
* Dropout: temp. rmv. units, at random, to prevent over-reliance

## Feed forward neural network

## recurrent neural network
neural network that generates output that feeds back into its own inputs

**pooling**
Reducing the size of input usig samples of regions of the input

**max pooling**
pooling by choosing the maximum value in each region

# Language
[overview](6-Language/6%20-%20Language.pdf)

* [Sentiment analysis](6-Language/demo/sentiment/sentiment.py)
* [Sentance parsing](6-Language/demo/cfg/cfg1.py)
* Translation
* [Sentence creation](6-language/demo/markov/generator.py)