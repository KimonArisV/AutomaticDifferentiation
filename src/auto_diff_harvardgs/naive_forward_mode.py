import numpy as np
import math


class Node():
    
    def __init__(self, val, deriv):
        self.val = val
        self.deriv = deriv
    
    def __str__(self):
        return f'Node: value={self.val}, derivative={self.deriv}'
        
    def get_value(self):
        return self.val
    
    def get_deriv(self):
        return self.deriv
    
    def __add__(self, other):
        try:
            val = self.val + other.val
            deriv = self.deriv + other.deriv
            return Node(val, deriv)
        except AttributeError:
            val = self.val + other
            deriv = self.deriv
            return Node(val, deriv)
    
    def __radd__(self, other):
        try:
            val = self.val + other.val
            deriv = self.deriv + other.deriv
            return Node(val, deriv)
        except AttributeError:
            val = self.val + other
            deriv = self.deriv
            return Node(val, deriv)
    
    def __sub__(self, other):
        try:
            val = self.val - other.val
            deriv = self.deriv - other.deriv
            return Node(val, deriv)
        except AttributeError:
            val = self.val - other
            deriv = self.deriv
            return Node(val, deriv)
    
    def __rsub__(self, other):
        try:
            val = other.val - self.val
            deriv = other.deriv - self.val
            return Node(val, deriv)
        except AttributeError:
            val = other - self.val
            deriv = - self.deriv
            return Node(val, deriv)
    
    def __mul__(self, other):
        try:
            val = self.val * other.val
            deriv = self.deriv*other.val + other.deriv*self.val
            return Node(val, deriv)
        except AttributeError:
            val = self.val * other
            deriv = self.deriv*other
            return Node(val, deriv)
    
    def __rmul__(self, other):
        try:
            val = self.val * other.val
            deriv = self.deriv*other.val + other.deriv*self.val
            return Node(val, deriv)
        except AttributeError:
            val = self.val * other
            deriv = self.deriv*other
            return Node(val, deriv)
        

    def __truediv__(self, other):
        try:
            val = self.val / other.val
            deriv = (self.deriv*other.val - other.deriv*self.val)/(other.val**2)
            return Node(val, deriv)
        except AttributeError:
            val = self.val / other
            deriv = self.deriv/other
            return Node(val, deriv)
    
    def __rtruediv__(self, other):
        try:
            val =  other.val/self.val
            deriv = (other.deriv*self.val - self.deriv*other.val )/(self.val**2)
            return Node(val, deriv)
        except AttributeError:
            val =  other/self.val
            deriv = other/self.deriv
            return Node(val, deriv)

    def __pow__(self, other):
        try:
            val =  self.val**other.val
            deriv = self.val**other.val*(other.val * self.deriv/self.val + other.deriv*np.log(self.val))
            return Node(val, deriv)
        except AttributeError:
            val =  self.val**other
            deriv = self.val**other*(other * self.deriv/self.val)
            return Node(val, deriv)
    
    def __rpow__(self, other):
        try:
            val =  other.val**self.val
            deriv = other.val**self.val*(self.val * other.deriv/other.val + self.deriv*np.log(other.val))
            return Node(val, deriv)
        except AttributeError:
            val =  other**self.val
            deriv = other**self.val*(self.deriv*np.log(other))
            return Node(val, deriv)

    def __neg__(self):
        val =  -self.val
        deriv = -self.deriv
        return Node(val, deriv)
    
    def __pos__(self):
        val =  self.val
        deriv = self.deriv
        return Node(val, deriv)


def make_dual_numbers(value, seed=[]):
    """Creates nodes for automatic differentation.
    Keyword arguments:
    value -- initial values Nodes to be created. Int, float, or arrays accepted
    seed -- initial seed values. len(seed) == len(value) if seed is given. If no seed given, default seed to 1.
    :return -- List of nodes or single node
    """
    output = []
    if isinstance(value, list):  # create multiple nodes
        if len(seed) == 0:
            seed = np.ones(len(value))
        if len(seed) != len(value):
            raise IndexError(f'lengths of values and seed must match')
        for idx, val in enumerate(value):
            s = np.zeros(len(value))
            s[idx] = seed[idx]
            output.append(Node(val, s))
        return output
    else:  # one node only
        return Node(value, seed)


def eval_vector(func_list):
    '''
    Evaluate multiple equations at once
    func_list -- list of equations
    :return: list of dictionaries containing value and derivatives
    '''
    output = []
    for eq in func_list:
        output.append({"value": eq.val, "derivative": eq.deriv})
    return output


def sin(obj):
    if type(obj) == Node:
        val = np.sin(obj.val)
        deriv = np.cos(obj.val) * obj.deriv
        return Node(val, deriv)
    else:
        return np.sin(obj)


def cos(obj):
    if type(obj) == Node:
        val = np.cos(obj.val)
        deriv = -np.sin(obj.val) * obj.deriv
        return Node(val, deriv)
    else:
        return np.cos(obj)


def tan(obj):
    if type(obj) == Node:
        val = np.tan(obj.val)
        deriv = (np.arccos(obj.val) ** 2) * obj.deriv
        return Node(val, deriv)
    else:
        return np.tan(obj)


def exp(obj):
    if type(obj) == Node:
        val = np.exp(obj.val)
        deriv = np.exp(obj.val) * obj.deriv
        return Node(val, deriv)
    else:
        return np.exp(obj)

