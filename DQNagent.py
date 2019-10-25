import numpy as np
import matplotlib.pyplot as plt
import textDisplay

#from game import Directions
from game import Agent

import NN_util
import copy


## Glorot
def init_NN_glorot_Tanh(L, uniform=False):
    """
    Initializer using the glorot initialization scheme
    """
    weights = []
    biases = []
    for i in range(len(L) - 1):
        if uniform:
            """ YOUR CODE HERE """
            bound = np.sqrt(6 / (L[i] + L[i + 1]))  # <- replace with proper initialization
            weights.append(np.random.uniform(low=-bound, high=bound, size=[L[i], L[i + 1]]))
            biases.append(np.random.uniform(low=-bound, high=bound, size=[1, L[i + 1]]))
        else:
            """ YOUR CODE HERE """
            std = 2 / (L[i] + L[i + 1])  # <- replace with proper initialization
            weights.append(np.random.normal(loc=0.0, scale=std, size=[L[i], L[i + 1]]))
            biases.append(np.random.normal(loc=0.0, scale=std, size=[1, L[i + 1]]))

    return (weights, biases)


## He
def init_NN_he_ReLU(L, uniform=False):
    """
    Initializer using the He initialization scheme
    """
    weights = []
    biases = []
    for i in range(len(L) - 1):
        if uniform:
            """ YOUR CODE HERE """
            bound = np.sqrt(6 / L[i])  # <- replace with proper initialization
            weights.append(np.random.uniform(low=-bound, high=bound, size=[L[i], L[i + 1]]))
            biases.append(np.random.uniform(low=-bound, high=bound, size=[1, L[i + 1]]))
        else:
            """ YOUR CODE HERE """
            std = 2 / L[i]  # <- replace with proper initialization
            weights.append(np.random.normal(loc=0.0, scale=std, size=[L[i], L[i + 1]]))
            biases.append(np.random.normal(loc=0.0, scale=std, size=[1, L[i + 1]]))

    return (weights, biases)

def init_NN_Glorot(L, activations, uniform=False):
    """
    Initializer using the glorot initialization scheme
    """

    weights = []
    biases = []
    """ YOUR CODE HERE!"""
    for i in range(len(L) - 1):
        # print(activations[i].__name__ == 'ReLU')
        if activations[i].__name__ == 'ReLU':
            w, b = init_NN_he_ReLU(L[i:i + 2])
        else:
            w, b = init_NN_glorot_Tanh(L[i:i + 2])
        weights += w
        biases += b

    return (weights, biases)



def backward_pass(x, t, y, z, a, NN, activations, loss):
    """
    This function performs a backward pass ITERATIVELY. It saves lists all of the derivatives in the process

    Input:
    x:           The input used for the batch                (np.array)
    t:           The observed targets                        (np.array, the first dimension must be the same to x)
    y:           The output of the forward_pass of NN for x  (np.array, must have the same shape as t)
    a:           The affine transforms from the forward_pass (np.array)
    z:           The activated units from the forward_pass (np.array)
    activations: The activations to be used                  (list of functions)
    loss:        The loss function to be used                (one function)

    Output:
    g_w: A list of gradients for every hidden unit
    g_b: A list of gradients for every bias

    Shapes for the einsum:
    b: batch size
    i: size of the input hidden layer (layer l)
    o: size of the output (layer l+1)
    """
    BS = x.shape[0]  # Implied batch shape

    # First, let's compute the list of derivatives of z with respect to a
    d_a = []
    for i in range(len(activations)):
        d_a.append(activations[i](a[i], derivative=True))

    # Second, let's compute the derivative of the loss function
    t = t.reshape(BS, -1)

    """ YOUR CODE HERE!"""
    d_loss = loss(t, y, derivative=True)  # <- Insert correct expression here

    # Third, let's compute the derivative of the biases and the weights
    g_w = []  # List to save the gradient of the weights
    g_b = []  # List to save the gradients of the biases

    delta = np.einsum('bo, bo -> bo', d_loss,
                      d_a[-1])  # loss shape: (b, o); pre-activation units shape: (b, o) hadamard product

    g_b.append(np.mean(delta, axis=0))
    g_w.append(
        np.mean(np.einsum('bo, bi -> bio', delta, z[-2]), axis=0))  # delta shape: (b, o), activations shape: (b, h)

    for l in range(1, len(NN[0])):
        d_L_d_z = np.einsum('bo, io -> bi', delta,
                            NN[0][-l])  # Derivative of the Cost with respect to an activated layer d_C_d_z.
        #  delta shape: as above; weights shape: (i, o)
        # Delta: d_C_d_z (element-wise mult) derivative of the activation layers
        #  delta shape: as above; d_z shape: (b, i)
        delta = np.einsum('bo, bo -> bo', d_L_d_z, d_a[-1 - 1])  # <- Insert correct expression

        g_b.append(np.mean(delta, axis=0))
        g_w.append(np.mean(np.einsum('bo, bi -> bio', delta, z[-l - 2]),
                           axis=0))  # Derivative of cost with respect to weights in layer l:
        # delta shape: as above; activations of l-1 shape: (b, i)

    return g_b[::-1], g_w[::-1]

class DQNagent(Agent):
    def __init__(self, **args):
        """ Initialize the neural network etc."""
        """ DON'T CHANGE THIS PART"""
        Agent.__init__(self, **args)
        self.verbose = False

        self.a_dict = NN_util.TwoWayDict()
        self.a_dict["North"] = 0
        self.a_dict["East"] = 1
        self.a_dict["South"] = 2
        self.a_dict["West"] = 3
        # self.a_dict["Stop"] = 4
        self.num_actions = len(self.a_dict)

        self.prev_state = None
        self.prev_action = None
        self.prev_score = 0.0
        self.exp = []

        """ PLAY AROUND AND CHANGE THIS PART"""
        self.eps = 0.2  # For epsilon greedy action selection.
        self.alpha = 1e-6  # learning rate
        self.gamma = 0.99  # discount factor

        self.layers = [2, 256, 256, 256, self.num_actions]
        self.activation_fns = [NN_util.ReLU, NN_util.ReLU, NN_util.ReLU, NN_util.Linear]
        assert len(self.layers) == len(self.activation_fns) + 1, "Number of layers and activation functions don't match!"

        """ DON'T CHANGE THIS PART"""
        self.NN = init_NN_Glorot(self.layers, self.activation_fns)

    def getNetworkInput(self, state):
        return np.asarray([state.getPacmanPosition()])

    def getReward(self, state):
        r = state.getScore() - self.prev_score
        self.prev_score = state.getScore()
        return r

    def updateNetwork(self):
        """ Update the network parameters"""
        s, a, r, s_, a_, d = self.exp[-1]  # CHANGE this for experience replay

        # Compute our estimate of the Q value, for updating the network
        """ YOUR CODE HERE! """
        # The Q-value in state s. Should be a [1, self.num_actions] shaped np.array

        # Compute the target
        # Forward pass
        ACT_F_UT = [NN_util.Linear, NN_util.Linear]
        test_a, test_z = NN_util.forward_pass(s, self.NN, ACT_F_UT)
        Q_pred = test_a[-1]
        """ YOUR CODE HERE! """
        # the target is observed reward + the discounted future reward (Q-value)
        # NB: at the terminal state, i.e. d==True the target is ONLY the reward.
        target = None
        # Update the network
        g_b, g_w = backward_pass(s, target, Q_pred, test_z, test_a,
                                 self.NN, self.activation_fns, NN_util.squared_error)

        # Stochastic gradient descent
        for l in range(len(g_b)):
            self.NN[0][l] -= self.alpha * g_w[l]
            self.NN[1][l] -= self.alpha * g_b[l]

    def terminal_update(self, state):
        s = self.getNetworkInput(state)
        r = self.getReward(state)
        a = None
        d = True

        self.exp.append([self.prev_state, self.prev_action, r, s, a, d])

        self.updateNetwork()
        self.save_weights()

        ## Clear everything, ready for next run!
        ## You will need to change this for experience replay
        """ YOUR CODE HERE! """
        self.prev_state = None
        self.prev_action = None
        self.prev_score = 0.0
        self.exp = []

    def getAction(self, state):
        s = self.getNetworkInput(state)
        r = self.getReward(state)

        if np.random.uniform() > self.eps:
            NN_forward = NN_util.forward_pass(s, self.NN, self.activation_fns)
            a = np.argmax(NN_forward[1][-1])
            if self.verbose:
                print(s, 'pick', self.a_dict[a])
        else:
            # Pick random legal action
            a = np.random.choice(range(self.num_actions))
            if self.verbose:
                print(s, 'random', self.a_dict[a])

        if self.a_dict[a] not in state.getLegalActions():
            # If illegal action is selected - set a negative reward
            # We also force the action to be 'Stop', but we do that in the
            # end of the function.
            r += -10
            if self.verbose:
                print('    ', 'illegal')

        if self.prev_state is not None:
            d = False  # whether or not we are terminal
            self.exp.append([self.prev_state, self.prev_action, r, s, a, d])
            self.updateNetwork()

        self.prev_state = s
        self.prev_action = a

        if self.a_dict[a] not in state.getLegalActions():
            return "Stop"

        return self.a_dict[a]

    def save_weights(self):
        # TODO
        pass

    def load_weights(self):
        # TODO
        pass


from pacman import *
import graphicsDisplay
if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python pacman.py

    See the usage string for more details.

    > python pacman.py --help
    """
    str_args = ['-l', 'TestMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    args = readCommand(str_args)
    args['display'] = textDisplay.NullGraphics()  # Disable rendering

    args['pacman'] = DQNagent()
    out = runGames(**args)

    scores = [o.state.getScore() for o in out]
    print(scores)

