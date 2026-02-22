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
        """Recursively counts the number of nodes (or just leaves) below this node."""
        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves=only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves=only_leaves)
        return count

    def left_child_add_prefix(self, text):
        """Adds formatting prefix for the left child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Adds formatting prefix for the right child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            # Use 7 spaces to pad below right children (no vertical connecting pipe)
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """Returns the string representation of the node and its children."""
        node_type = "root" if self.is_root else "-> node"
        res = f"{node_type} [feature={self.feature}, threshold={self.threshold}]\n"
        
        if self.left_child:
            res += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            res += self.right_child_add_prefix(str(self.right_child))
            
        return res.rstrip()


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
        """A leaf always counts as 1 node."""
        return 1

    def __str__(self):
        """Returns the string representation of the leaf."""
        return (f"-> leaf [value={self.value}]")


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
        """Counts the total number of nodes or leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns the string representation of the entire tree."""
        return self.root.__str__()
