import numpy as np
import math
import heapq


def sin(obj):
    '''Takes Node type object as input. Returns Node with sin func'''
    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.sin_eval, left=obj, right=None)
    return root.check_existence()


def cos(obj):
    '''Takes Node type object as input. Returns Node with cos func'''
    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.cos_eval, left=obj, right=None)
    return root.check_existence()


def tan(obj):
    '''Takes Node type object as input. Returns Node with tan func'''
    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.tan_eval, left=obj, right=None)
    return root.check_existence()


def exp(obj):
    '''Takes Node type object as input. Returns Node with e^x func'''
    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.exp_eval, left=obj, right=None)
    return root.check_existence()


def arcsin(obj):
    '''Takes Node type object as input. Returns Node with arcsin func'''

    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.arcsin_eval, left=obj, right=None)
    return root.check_existence()


def arccos(obj):
    '''Takes Node type object as input. Returns Node with arccos func'''

    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.arccos_eval, left=obj, right=None)
    return root.check_existence()


def arctan(obj):
    '''Takes Node type object as input. Returns Node with arctan func'''

    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.arctan_eval, left=obj, right=None)
    return root.check_existence()


def sinh(obj):
    '''Takes Node type object as input. Returns Node with sinh func'''

    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.mult_eval, left=exp(obj) - exp(-obj), right=Node.nodify(2) ** -1)
    return root.check_existence()


def cosh(obj):
    '''Takes Node type object as input. Returns Node with cosh func'''

    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.mult_eval, left=exp(obj) + exp(-obj), right=Node.nodify(2) ** -1)
    return root.check_existence()


def tanh(obj):
    '''Takes Node type object as input. Returns Node with tanh func'''

    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.mult_eval, left=sinh(obj), right=cosh(obj) ** -1)
    return root.check_existence()


def logistic(obj):
    '''Takes Node type object as input. Returns Node with logistics func'''
    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.mult_eval, left=Node.nodify(1), right=(1 + exp(obj)) ** -1)
    return root.check_existence()


def sqrt(obj):
    '''Takes Node type object as input. Returns Node with the square root function'''
    if type(obj) != Node:
        obj = Node.nodify(obj)

    root = Node(func=Node.pow_eval, left=obj, right=Node.nodify(1 / 2))
    return root.check_existence()


def log(obj, base=np.exp(1)):
    '''Takes Node type object as input. Returns Node with log method

    :param obj: Node
    :param base: int or float (default is natural log base e)
    '''
    if type(obj) != Node:
        obj = Node.nodify(obj)

    if type(base) != Node:
        base = Node.nodify(base)

    root = Node(func=Node.log_eval, left=obj, right=base)
    return root.check_existence()


class Node():
    '''A single class implemeting Automatic Differentiation.

    Objects of Node() class can be used to define fuctions, evaluate the function with given values
    and calculate the derivatives with given seeds. Instances of Node() class should be created by calling Node().create_nodes(n:int) function. This
    function returns n variables of type Node(), which can be used to build a function. After a function, f:Node, is defined the user can call
    f.evaluate(values, seeds) which will return value and derivatives. evaluate() can be called mutliple times with different values.

    Example:
    a, b, c = Node.create_nodes(3)

    d = (sin(a) + b)**2
    e = 3 + c + log(b, 2)

    f = [d, e]
    result = Node.evaluate(f, [np.pi,2, 4], [1, 1, 1])

    result:
        (array([4., 8.]),
        array([[-4.        ,  4.        ,  0.        ],
        [ 0.        ,  0.72134752,  1.        ]]))

    The result of the function is tuple of two arrays. The first one retuns the value of the two functions,
    the second array holds an array of gradients at a given seed.

    Note:
    The order of the elemnets in the list passed as arguments: values and seeds to evaluate()
    corresponds to the order to the nodes retuned by the call Node.create_nodes(n).
    '''
    # Tree build
    _node_register = {}
    # eval level
    _eval_register = {}
    _n_vars = 0

    def __init__(self, val=None, deriv=None, right=None, left=None, func=None, hash_=None):
        '''Class initailization function

        Parameters:
        val (int): value of the node
        deriv (int): seed of the node
        right (Node): right child of the Node
        left (Node): left child of the Node
        func (Node.func): evaluation function to be called during evaluation
        hash_ (str): Node's unique identifier

        An object of type Node can be initialised without passing some or all the values.
        '''
        self.val = val
        self.deriv = deriv
        self.left = left
        self.right = right
        self.func = func
        self.hash_ = hash_

    def __str__(self):
        '''Represents a Node as a string'''
        pretty_func = 'None' if not self.func else str(self.func).split()[1]
        pretty_str = f'Node(val:{self.val}, deriv:{self.deriv}, func:{pretty_func}, hash:{self.hash_})\n' + \
                     '\n|\n|-(L)->' + '\n|      '.join(str(self.left).split('\n')) + \
                     '\n|\n|-(R)->' + '\n|      '.join(str(self.right).split('\n'))
        return pretty_str

    @staticmethod
    def create_nodes(n: int):
        '''Takes input n number of variables. Returns single variable (n=1) or list of n variables as type Node'''
        Node._node_register = {}
        Node._n_vars = n
        for i in range(0, n):
            # hashes for variables are letters a,b,c...
            hash_ = hash(chr(97 + i))
            Node._node_register[hash_] = Node(hash_=hash_)
        if n == 1:
            return Node._node_register[hash_]
        else:
            return list(Node._node_register.values())

    @staticmethod
    def set_values(values, seeds):
        '''Input lists of values and list of seeds to store in the node register. Returns None'''
        assert len(values) == len(seeds), Exception('The two inputs must be of the same size.')
        for i in range(0, len(values)):
            hash_ = hash(chr(97 + i))
            Node._node_register[hash_].val = values[i]
            Node._node_register[hash_].deriv = np.zeros(len(values))
            if seeds[i] != 0:
                Node._node_register[hash_].deriv[i] = seeds[i]

    @staticmethod
    def evaluate(node, values, seeds=[]):
        ''' Evalutes and calculates derivatives of equation or vectors of equations
        :param node: equation or list of equations
        :param values: list of values to evaluate function at
        :param seeds: list of seeds to evaluate derivative at (optional, default set to 1)
        :return: list containing evaluated val, partial derivatives
        '''

        assert Node._n_vars == len(values), f'Length of values must be equal to number of variables'
        assert isinstance(node, Node) or isinstance(node, list), f'Input {node} must be type Node or list'

        if len(seeds) == 0:
            seeds = np.ones(Node._n_vars)
        assert len(seeds) == len(values), 'Length of seeds must be equal to length of values'

        Node._eval_register = {}
        Node.set_values(values, seeds)

        if isinstance(node, Node):
            node.eval_node()
            return node.val, node.deriv

        vals, derivs = [], []
        for eq in node:
            eq.eval_node()
            vals.append(eq.val)
            derivs.append(eq.deriv)
        return vals, derivs

    def eval_node(self):
        '''Takes self as an argument ensures that its descendants are evaluated'''

        if self.left is not None and self.left.func != None:
            self.left.eval_node()
        if self.right is not None and self.right.func != None:
            self.right.eval_node()
        self.func(self)
        return

    @staticmethod
    def nodify(other):
        '''If other is not already a Node, creates other'''
        if type(other) == Node:
            other_node = other
        else:
            # hashes for constants are calculated by hashing the string value of them
            other_node = Node(val=other, deriv=0, hash_=hash(str(other)))
            other_node.set_hash()
        return other_node

    def check_existence(self):
        '''
        returns itself if no equivalent node exist in the graph, if not it returns itself.

        Calculate and store the hash of the node, if the node does not have one already.

        If an equivalent node already exists (checked with the use of node.hash),
        return the node. Otherise, add the node to the node_register.

        '''
        self.set_hash()

        if self.hash_ in Node._node_register:
            return Node._node_register[self.hash_]
        Node._node_register[self.hash_] = self
        return self

    def set_hash(self):
        '''
        Sets hash based the hash of its descendants and its own operation.
        This hash is deterministic and allows for finding previously encountered nodes and subgraphs that are equivalent.
        This allows for optimizing the graph to avoid performing the same operations.
        The operations (functions) are converted to strings which are cut off after 22 characters to avoid the memory part of the string
        The hash is set based on 3 main cases, and if the hash already exist it is returned.
        1) The operation of the node is unary and the hash is only set based on its own operation and left child
        2) The operation is non-commutative and the hash is set based on its two children, and its operator.
        3) The operation is commutative and the hash is set based on the entire commutative subgraph and the operator.

        '''
        unary_func = [
            Node.exp_eval, Node.neg_eval,
            Node.tan_eval, Node.cos_eval, Node.sin_eval,
            Node.arcsin_eval, Node.arccos_eval, Node.arctan_eval
        ]

        noncommut_func = [Node.pow_eval, Node.log_eval]

        # if hash already exists, return
        if self.hash_ is not None:
            return

        # Create a hash if one does not already exist
        hash_ = []
        # Case1: single node function
        if self.func in unary_func:
            hash_.append(self.left.hash_)
            hash_.append(str(self.func)[:22])
            self.hash_ = hash(tuple(hash_))
            return
        # Case2: double node non-commutative function
        if self.func in noncommut_func:
            hash_.append(self.left.hash_)
            hash_.append(self.right.hash_)
            hash_.append(str(self.func)[:22])
            self.hash_ = hash(tuple(hash_))
            return

        # Case3: double node commutative function
        # 1. need to order children hashes, as a*b == b*a
        min_hash = min(self.left.hash_, self.right.hash_)
        first = self.left if self.left.hash_ == min_hash else self.right
        second = self.left if self.left.hash_ != min_hash else self.right
        # 2. If any of the children was a node same operator we need to pass the list of
        #   hashes of the descendants in order to be equate a*b*c to c* a * b
        first_hash = first.hash_list if first.func == self.func else [first.hash_]
        second_hash = second.hash_list if second.func == self.func else [second.hash_]
        # merge sort the hashes
        self.hash_list = list(heapq.merge(first_hash, second_hash))
        self.hash_ = hash(tuple(self.hash_list + [str(self.func)[:22]]))

    # Addition
    def __add__(self, other):
        '''overload addition'''
        other_node = Node.nodify(other)
        root = Node(func=Node.add_eval, left=self, right=other_node)
        return root.check_existence()

    def __radd__(self, other):
        '''overload addition'''
        other_node = Node.nodify(other)

        root = Node(func=Node.add_eval, left=other_node, right=self)
        return root.check_existence()

    @staticmethod
    def add_eval(node):
        '''Evaluate addition of nodes'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right != None), 'Addition needs two nodes.'
            node.val = node.left.val + node.right.val
            node.deriv = node.left.deriv + node.right.deriv
            Node._eval_register[node.hash_] = 1

    # Subtraction
    def __sub__(self, other):
        '''overload subtraction'''
        other_node = Node.nodify(other)
        root = Node(func=Node.add_eval, left=self, right=-other_node)
        return root.check_existence()

    def __rsub__(self, other):
        '''overload subtraction'''
        other_node = Node.nodify(other)
        root = Node(func=Node.add_eval, left=other_node, right=-self)
        return root.check_existence()

    def __neg__(self):
        '''overload negation'''
        root = Node(func=Node.neg_eval, left=self)
        return root.check_existence()

    @staticmethod
    def neg_eval(node):
        '''Evaluate subtraction of nodes'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None)
            node.val = -node.left.val
            node.deriv = -node.left.deriv
            Node._eval_register[node.hash_] = 1

    # multiplication
    def __mul__(self, other):
        '''overload multiplication'''
        other_node = Node.nodify(other)
        root = Node(func=Node.mult_eval, left=self, right=other_node)
        return root.check_existence()

    def __rmul__(self, other):
        '''overload multiplication'''
        other_node = Node.nodify(other)
        root = Node(func=Node.mult_eval, left=other_node, right=self)
        return root.check_existence()

    @staticmethod
    def mult_eval(node):
        '''Evaluate multiplication of nodes'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right != None), 'Multiplication needs two nodes.'
            node.val = node.left.val * node.right.val
            node.deriv = node.left.deriv * node.right.val + node.left.val * node.right.deriv
            Node._eval_register[node.hash_] = 1

    # division
    def __truediv__(self, other):
        '''overload division'''

        other_node = Node.nodify(other)
        root = Node(func=Node.mult_eval, left=self, right=other_node ** -1)
        return root.check_existence()

    def __rtruediv__(self, other):
        '''overload division'''
        other_node = Node.nodify(other)
        root = Node(func=Node.mult_eval, left=other_node, right=self ** -1)
        return root.check_existence()

    def __pow__(self, other):
        '''overload power'''
        other_node = Node.nodify(other)
        root = Node(func=Node.pow_eval, left=self, right=other_node)
        return root.check_existence()

    def __rpow__(self, other):
        '''overload power'''
        other_node = Node.nodify(other)
        root = Node(func=Node.pow_eval, left=other_node, right=self)
        return root.check_existence()

    @staticmethod
    def pow_eval(node):
        '''Evaluate power function'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right != None), 'power function needs two nodes.'
            node.val = node.left.val ** node.right.val
            node.deriv = node.left.val ** node.right.val * (
                    node.right.val * node.left.deriv / node.left.val + node.right.deriv * np.log(node.left.val))
            Node._eval_register[node.hash_] = 1

    @staticmethod
    def sin_eval(node):
        '''Evaluate sin function'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right == None), 'sin function needs one child'
            node.val = np.sin(node.left.val)
            node.deriv = np.cos(node.left.val) * node.left.deriv
            Node._eval_register[node.hash_] = 1

    @staticmethod
    def cos_eval(node):
        '''Evaluate cos function'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right == None), 'cos function needs one child'

            node.val = np.cos(node.left.val)
            node.deriv = -np.sin(node.left.val) * node.left.deriv
            Node._eval_register[node.hash_] = 1

    @staticmethod
    def tan_eval(node):
        '''Evaluate tan function'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right == None), 'tan function needs one child'

            node.val = np.tan(node.left.val)
            node.deriv = (np.cos(node.left.val) ** -2) * node.left.deriv
            Node._eval_register[node.hash_] = 1

    @staticmethod
    def exp_eval(node):
        '''Evaluate e^x function'''

        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right == None), 'exp function needs one child'

            node.val = np.exp(node.left.val)
            node.deriv = np.exp(node.left.val) * node.left.deriv
            Node._eval_register[node.hash_] = 1

    @staticmethod
    def arcsin_eval(node):
        '''Evaluate arcsin function'''

        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right == None), 'arcsin function needs one child'
            node.val = np.arcsin(node.left.val)
            node.deriv = node.left.deriv / (np.sqrt(1 - node.left.val ** 2))
            Node._eval_register[node.hash_] = 1

    @staticmethod
    def arccos_eval(node):
        '''Evaluate arccos function'''

        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right == None), 'arccos function needs one child'
            node.val = np.arccos(node.left.val)
            node.deriv = - node.left.deriv / (np.sqrt(1 - node.left.val ** 2))
            Node._eval_register[node.hash_] = 1

    @staticmethod
    def arctan_eval(node):
        '''Evaluate arctan function'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right == None), 'arctan function needs one child'
            node.val = np.arctan(node.left.val)
            node.deriv = node.left.deriv / (1 + node.left.val ** 2)
            Node._eval_register[node.hash_] = 1

    @staticmethod
    def log_eval(node):
        '''Evaluate log function'''
        if not node.hash_ in Node._eval_register:
            assert (node.left != None) and (node.right != None), 'Log needs two nodes.'
            node.val = math.log(node.left.val, node.right.val)
            node.deriv = node.left.deriv / (node.left.val * math.log(node.right.val))
            Node._eval_register[node.hash_] = 1

    def __eq__(self, other):
        '''overload equality to compare Nodes'''
        if type(other) != Node:
            return False
        else:
            return self.hash_ == other.hash_

    def __neq__(self, other):
        '''overload neq to compare Nodes'''
        return not (self == other)
