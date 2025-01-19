import numpy as np
from numpy.linalg import inv
import math
from scipy import special


class BDeuScore:
    def __init__(self, dataset, variables, sample_prior=1,
                 structure_prior=1):
        """Initialize the BDeuScore object.

        :param dataset: 2D np.array, with a row for each data
                        point and a column for each variable.
        :param variables: np.array containing the variable ids
        :param sample_prior: sample prior
        :param structure_prior: structure prior
        """

        assert len(dataset.shape) == 2, "Dataset must be 2-dimensional."
        # defining attributes
        self.dataset = dataset
        self.variables = variables
        self.sampleSize = dataset.shape[0]

        # note that the sample prior is set by the
        # sparsity parameter in runner.py
        self.sample_prior = sample_prior
        self.structure_prior = structure_prior

        # num_categories is always 2 for binary data
        # (TODO: if data is not binary we must create
        # a method which computes the number
        # of categories for each variable)
        self.num_categories = np.full(len(self.variables), 2)

    def local_score(self, node, parents):
        """
        Method to compute the score associated to having
        a certain set of parents associated to a node.

        :param node: int representing the node in question.
        :param parents: np.array of ints representing the parent nodes.
        """
        # number of categories of this node
        c = self.num_categories[node]

        # number of categories of the parent nodes
        dims = self.num_categories[parents]

        # number of parent states (i.e. multiplying the number of
        # categories of each variable together)
        r = np.prod(dims)

        # conditional cell coeffs for node given parents (node)
        n_jk = np.zeros((r, c))
        n_j = np.zeros(r)
        my_parents = self.dataset[:, parents]
        my_child = self.dataset[:, node]

        # populate the conditional cell coeffs
        for i in range(self.sampleSize):
            parent_values = my_parents[i]
            child_value = my_child[i]
            row_index = self.get_row_index(dims, parent_values)

            n_jk[row_index][child_value] += 1
            n_j[row_index] += 1

        # finally compute the score
        score = self.get_prior_for_structure(len(parents))

        cell_prior = self.sample_prior / (c*r)
        row_prior = self.sample_prior / r

        for j in range(r):
            score -= special.loggamma(row_prior + n_j[j])
            for k in range(c):
                score += special.loggamma(cell_prior + n_jk[j][k])

        score += r * special.loggamma(row_prior)
        score -= c * r * special.loggamma(cell_prior)

        return score

    def get_prior_for_structure(self, num_parents):
        """
        Method that returns the initial score value
        as defined by the structure prior.

        :param num_parents: int representing the number of
                            nodes in the parents np.array.
        """
        e = self.structure_prior
        vm = self.dataset.shape[0] - 1
        return num_parents*np.log(e/vm) + (vm - num_parents) * np.log(1-(e/vm))

    def get_row_index(self, dims, values):
        """
        Method that returns the index to increment in
        the cell coeffs.

        :param dims: np.array containing the number of categories
                     adopted by each node in the parents np.array.
        :param values: value of parent nodes at a specific time instant.
        """
        row_index = 0
        for i in range(len(dims)):
            row_index *= dims[i]
            row_index += values[i]
        return row_index

    def local_score_diff_parents(self, node1, node2, parents):
        """
        Method to compute the change in score resulting
        from adding node1 to the list of parents.

        :param node1: int representing the node to add
                      to list of parents.
        :param node2: int representing the node in question.
        :param parents: list of ints representing the parent nodes.
        """
        return self.local_score(node2, parents + [node1]) - self.local_score(node2, parents)

    def local_score_diff(self, node1, node2):
        """
        Method to compute the change in score resulting
        from having node1 as a parent.

        :param node1: int representing the parent node.
        :param node2: int representing the node in question.
        """
        return self.local_score_diff_parents(node1, node2, [])
