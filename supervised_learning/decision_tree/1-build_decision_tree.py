#!/usr/bin/env python3
"""
Module for building a Decision Tree.
Contains the Node, Leaf, and Decision_Tree classes.
"""
import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth of the tree below the current node."""
        max_d = self.depth
        if self.left_child:
            max_d = max(max_d, self.left_child.max_depth_below())
        if self.right_child:
            max_d = max(max_d, self.right_child.max_depth_below())
        return max_d

    def count_nodes_below(self, only_leaves=False):
        """
        Recursively counts the number of nodes (or just leaves) below this node.
        """
        # If we only want leaves, this internal node doesn't count as 1.
        count = 0 if only_leaves else 1
        
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves=only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves=only_leaves)
            
        return count


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf node."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        A leaf always counts as 1 node, whether we are counting all nodes or just leaves.
        """
        return 1


class Decision_Tree():
    """
    Represents a decision tree model.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Calculates the maximum depth of the entire decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the total number of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
