import torch
from itertools import combinations
from torch.nn import SELU, Linear, ModuleList, Module


class FETANetwork(Module):
    def __init__(self, n_features, n_hidden_layers=2, n_hidden_units=8, activation=SELU(), n_context_features=0):
        """
        Creates an instance of the FETA network.

        :param n_features: Size of the feature vectors
        :param n_hidden_layers: Number of hidden layers. Default: 2
        :param n_hidden_units: Number of hidden units. Default: 8
        :param activation: Activation function to be used. Default: SELU
        :param n_context_features: Number of additional context features. Default: 0 (no additional context information)
        """
        super(FETANetwork, self).__init__()
        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.activation = activation
        self.n_context_features = n_context_features

        # Create 0-order network
        self.hidden_layers_zeroth = []
        for i in range(n_hidden_layers):
            if i == 0:
                layer = Linear(n_features + n_context_features, n_hidden_units)
            else:
                layer = Linear(n_hidden_units, n_hidden_units)
            self.hidden_layers_zeroth.append(layer)
        self.scorer_zeroth = Linear(n_hidden_units, 1)

        # Create 1-order network
        self.hidden_layers_first = []
        for i in range(n_hidden_layers):
            if i == 0:
                layer = Linear(2 * n_features, n_hidden_units)
            else:
                layer = Linear(n_hidden_units, n_hidden_units)
            self.hidden_layers_first.append(layer)
        self.scorer_first = Linear(2 * n_hidden_units, 1)

        # Convert lists to module lists
        self.hidden_layers_zeroth = ModuleList(self.hidden_layers_zeroth)
        self.hidden_layers_first = ModuleList(self.hidden_layers_first)

        # Initialize weigths
        self.reset_weights()

    def forward(self, x, image_contexts=None):
        """
        Forwards the given feature vectors through the network in order to obtain feature vectors.

        :param x: Object feature vectors
        :param image_contexts: Additional representative of the context
        :return: Utility scores
        """
        n_objects = x.size()[1]

        # Compute 0th-order scores
        if image_contexts is not None:
            # Consider image context for 0th-order model
            image_contexts = image_contexts.unsqueeze(1)
            image_contexts = image_contexts.expand(image_contexts.size(0), n_objects, -1)
            zeroth_order_input = torch.cat((x, image_contexts), dim=2)
        else:
            zeroth_order_input = x

        zeroth_order_scores = self.get_zeroth_order_scores(zeroth_order_input, n_objects)

        # Compute pairwise scores
        outputs = [list() for _ in range(n_objects)]
        for i, j in combinations(range(n_objects), 2):
            x1 = x[:, i]
            x2 = x[:, j]
            x1x2 = torch.cat([x1, x2], dim=1)
            x2x1 = torch.cat([x2, x1], dim=1)

            for k in range(self.n_hidden_layers):
                x1x2 = self.activation(self.hidden_layers_first[k](x1x2))
                x2x1 = self.activation(self.hidden_layers_first[k](x2x1))

            merged_left = torch.cat([x1x2, x2x1], dim=1)
            merged_right = torch.cat([x2x1, x1x2], dim=1)

            score_left = self.scorer_first(merged_left)
            score_right = self.scorer_first(merged_right)

            outputs[i].append(score_left)
            outputs[j].append(score_right)

        outputs = [torch.cat(x, dim=1) for x in outputs]
        scores = torch.stack(outputs, dim=1)
        scores = torch.mean(scores, dim=2)

        # Add up 0-order and 1-order scores
        scores = torch.add(scores, zeroth_order_scores)

        return scores

    def get_zeroth_order_scores(self, x, n_objects):
        """
        Compute the scores of the 0-order model.

        :param x: Object feature vectors
        :param n_objects: Number of objects in the image
        :return: 0-order scores
        """
        scores = []
        for i in range(n_objects):
            score = x[:, i]
            for j in range(self.n_hidden_layers):
                score = self.activation(self.hidden_layers_zeroth[j](score))
            score = self.activation(self.scorer_zeroth(score))
            scores.append(score)
        tensor = torch.cat(scores, dim=1)
        return tensor

    def reset_weights(self):
        """
        Resets the layer weights.

        """
        for layer in self.hidden_layers_zeroth:
            torch.nn.init.xavier_uniform_(layer.weight)
        for layer in self.hidden_layers_first:
            torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.xavier_uniform_(self.scorer_zeroth.weight)
        torch.nn.init.xavier_uniform_(self.scorer_first.weight)
