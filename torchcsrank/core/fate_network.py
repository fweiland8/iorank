import torch
from torch.nn import Module, Linear, ModuleList, SELU


class FATENetwork(Module):
    def __init__(self, n_features, n_hidden_joint_layers=32, n_hidden_joint_units=32, n_hidden_set_layers=2,
                 n_hidden_set_units=2, activation=SELU(), n_context_features=0):
        """
        Creates an instance of the FATE network.

        :param n_features: Size of the feature vectors
        :param n_hidden_joint_layers: Number of hidden joint layers. Default: 2
        :param n_hidden_joint_units: Number of units in the hidden joint layers. Default: 32
        :param n_hidden_set_layers: Number of hidden set layers. Default: 2
        :param n_hidden_set_units: Number of units in the hidden set layers. Default: 32
        :param activation: Activation function to be used. Default: SELU
        :param n_context_features: Number of additional context features. Default: 0 (no additional context information)
        """
        super(FATENetwork, self).__init__()
        self.n_hidden_set_layers = n_hidden_set_layers
        self.n_hidden_set_units = n_hidden_set_units
        self.n_hidden_joint_layers = n_hidden_joint_layers
        self.n_hidden_joint_units = n_hidden_joint_units
        self.activation = activation
        self.n_context_features = n_context_features

        # Construct hidden joint layers
        self.joint_layers = []
        for i in range(n_hidden_joint_layers):
            if i == 0:
                layer = Linear(n_features + n_hidden_set_units + n_context_features, n_hidden_joint_units)
            else:
                layer = Linear(n_hidden_joint_units, n_hidden_joint_units)
            self.joint_layers.append(layer)

        # Construct score layer
        self.scorer = Linear(n_hidden_joint_units, 1)

        # Create set layer
        self.set_layers = []
        for i in range(n_hidden_set_layers):
            if i == 0:
                layer = Linear(n_features, n_hidden_set_units)
            else:
                layer = Linear(n_hidden_set_units, n_hidden_set_units)
            self.set_layers.append(layer)

        # Convert lists to module lists
        self.joint_layers = ModuleList(self.joint_layers)
        self.set_layers = ModuleList(self.set_layers)

        # Init weights
        self.reset_weights()

    def forward(self, x, image_contexts=None):
        """
        Forwards the given feature vectors through the network in order to obtain feature vectors.
        
        :param x: Object feature vectors
        :param image_contexts: Additional representative of the context
        :return: Utility scores
        """

        n_objects = x.size()[1]

        # Get feature representation of the set
        context_representation = self._get_context_representation(x, n_objects, image_contexts)

        scores = []
        for i in range(n_objects):
            obj = x[:, i]
            score = torch.cat((context_representation, obj), dim=1)
            for j in range(self.n_hidden_joint_layers):
                score = self.activation(self.joint_layers[j](score))
            score = self.activation(self.scorer(score))
            scores.append(score)
        return torch.cat(scores, dim=1)

    def _get_context_representation(self, x, n_objects, image_contexts=None):
        """
        Constructs a representative for the context.

        Includes both information about the input set (set context) as well as additional context information
        that can be provided as optional arguments.

        :param x: Object feature vectors
        :param n_objects: Number of objects in the image
        :param image_contexts: Additional representative of the context
        :return: Representation of the context
        """
        set_representation = self._get_set_representation(x, n_objects)
        if image_contexts is None:
            return set_representation
        else:
            return torch.cat((set_representation, image_contexts), 1)

    def _get_set_representation(self, x, n_objects):
        """
        Computes a representative for the input set

        :param x: Object feature vectors
        :param n_objects: Number of objects in the image
        :return: Input set representative
        """
        # Compute representative for each object
        representatives = []
        for i in range(n_objects):
            result = x[:, i]
            for j in range(self.n_hidden_set_layers):
                result = self.activation(self.set_layers[j](result))
            representatives.append(result)
        # Sum representatives up in tensor
        tensor = torch.stack(representatives, dim=1)
        return tensor.mean(1)

    def reset_weights(self):
        """
        Resets the layer weights.

        """
        for layer in self.joint_layers:
            torch.nn.init.xavier_uniform_(layer.weight)
        for layer in self.set_layers:
            torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.xavier_uniform_(self.scorer.weight)
