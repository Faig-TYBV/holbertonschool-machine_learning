#!/usr/bin/env python3
'''Adding max_depth_below'''


import numpy as np
'''Importing numpy library'''


class Node:
    '''Node class'''

    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        '''initializing fields'''

        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        '''finding max depth'''

        return max(self.left_child.max_depth_below(), self.right_child.max_depth_below())
    
    def count_nodes_below(self, only_leaves=False):
        '''counting nodes below'''

        if only_leaves:
            return self.left_child.count_nodes_below(only_leaves) + self.right_child.count_nodes_below(only_leaves)
        else:
            return 1 + self.left_child.count_nodes_below(only_leaves) + self.right_child.count_nodes_below(only_leaves)
        
    def __str__(self) :
        '''nodes in a printable format'''
        
        self_line = ""
        if self.is_root:
            self_line = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            self_line = f"-> node [feature={self.feature}, threshold={self.threshold}]"
        left_text = self.left_child.__str__()
        right_text = self.right_child.__str__()
        left_text = self.left_child_add_prefix(left_text)
        right_text = self.right_child_add_prefix(right_text)
        return self_line + "\n" + left_text + right_text
        
    def left_child_add_prefix(self,text):
            '''left part of the node'''

            lines=text.split("\n")
            new_text="    +--"+lines[0]+"\n"
            for x in lines[1:-1] :
                new_text+=("    |  "+x)+"\n"
            return (new_text)
    
    def right_child_add_prefix(self, text):
            '''right part of the node'''

            lines=text.split("\n")
            new_text="    +--"+lines[0]+"\n"
            for x in lines[1:-1] :
                new_text+=("       "+x)+"\n"
            return (new_text)
    
    def get_leaves_below(self):
        '''returning leaves locating below'''

        left_child_leaves = self.left_child.get_leaves_below() if self.left_child else []
        right_child_leaves = self.right_child.get_leaves_below() if self.right_child else []
        res = left_child_leaves + right_child_leaves
        return res
    
    def update_bounds_below(self) :
        if self.is_root : 
            self.upper = { 0:np.inf }
            self.lower = {0 : -1*np.inf }

        for child in [self.left_child, self.right_child] :
            # To Fill : compute and attach the lower and upper dictionaries to the children
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

        # Left child: feature <= threshold → tighten upper bound
        self.left_child.upper[self.feature] = self.threshold
        # Right child: feature > threshold → tighten lower bound
        self.right_child.lower[self.feature] = self.threshold
        for child in [self.left_child, self.right_child] :
            child.update_bounds_below()

class Leaf(Node):
    '''Leaf class extending Node class'''

    def __init__(self, value, depth=None):
        '''initializing fields'''

        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self) :
        '''returning depth'''

        return self.depth
    
    def count_nodes_below(self, only_leaves=False) :
        '''counting nodes'''

        return 1
    
    def __str__(self):
        '''leaf node in a printable format'''
    
        return (f"-> leaf [value={self.value}]")
    
    def get_leaves_below(self) :
        """returning the leaf"""
    
        return [self]
    
    def update_bounds_below(self) :
        '''there is nothing below so just pass'''    
        
        pass 

class Decision_Tree():
    '''Decision Tree class'''

    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        '''initializing fields'''

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

    def depth(self) :
        '''finding depth'''

        return self.root.max_depth_below()
    
    def count_nodes(self, only_leaves=False):
        '''counting nodes'''

        return self.root.count_nodes_below(only_leaves=only_leaves)
    
    def __str__(self):
        '''decision tree in printable format'''
        
        return self.root.__str__()
    
    def get_leaves(self) :
        """returning the leaves locating below"""
    
        return self.root.get_leaves_below()
    
    def update_bounds(self) :
        '''updating bounds starting with the root node'''

        self.root.update_bounds_below() 