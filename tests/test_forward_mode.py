#!/usr/bin/env python
# coding: utf-8


#import libraries 
import pytest # python standard library
import unittest
import numpy as np
import sys
#syntax needed and import of your package with the forward mode
sys.path.append("..")
from src.auto_diff_harvardgs.forward_mode import *


#test the calling of functions and the difference from the expected outcome    
class TestFunctions(unittest.TestCase):
        def test_object_type(self):
                a = Node(val=1,deriv=1)
                assert type(tan(math.pi))==Node
                assert type(arcsin(math.pi))==Node
                assert type(arccos(math.pi))==Node
                assert type(tanh(a))==Node
                assert type(sinh(math.pi))==Node
                assert type(cosh(math.pi))==Node
                assert type(exp(math.pi))==Node
                assert type(log(math.pi))==Node
                assert type(str(sin(math.pi / 4)))==str

class TestFunctions(unittest.TestCase):
        def test_val(self):
                a = Node(val=1,deriv=1)
                self.assertEqual(a.val, 1)
                
        def test_mult_input_output(self):
                # Create variables
                x, y = Node.create_nodes(2)
                # Define equation, variable values and seed
                f1 = exp(-(sin(x) - cos(y)) ** 2)
                f2 = exp(x+y)
                values = [math.pi / 2, math.pi / 3]
                seeds = [1, 1]
                # Evaluate
                res1 = Node.evaluate(f1, values, seeds) # single function
                res2 = Node.evaluate([f1, f2], values, seeds) #multi function
                assert (res1[0]==(res2[0][0]))
                assert (res1[1]-(res2[1][0])).all()<=np.finfo(float).eps
                
        def test_eq_neq(self):
                assert sinh(math.pi / 4)!=(cosh(math.pi / 4))
                assert arcsin(math.pi / 4)==(arcsin(math.pi / 4))
                
        def test_pow_sin_cos(self):
                values = [math.pi / 2, math.pi / 3]
                seeds = [1, 1]
                x, y =Node.create_nodes(2)
                f = exp(-(sin(x) - cos(y)) ** 2)
                res= Node.evaluate(f, values, seeds) # single function
                assert (res[0]- np.exp(-0.25))<=np.finfo(float).eps
                assert (res[1][0]- 0)<=np.finfo(float).eps
                assert (res[1][1]- (-math.sqrt(3)/2*np.exp(-0.25)))<=np.finfo(float).eps
                
        def test_arcsin_arccos_log(self):
                # Create variables
                x, y = Node.create_nodes(2)
                # Define equation, variable values and seed
                f1 = tan(-(arcsin(x) - arccos(y)))
                f2 = log(x/2+y)
                values = [0.1, .1]
                seeds = [1, 1]
                # Evaluate
                res1 = Node.evaluate(f1, values, seeds) # single function
                res2 = Node.evaluate([f1, f2], values, seeds) #multi function
                assert (res1[0]==(res2[0][0]))
                assert (res1[1]-(res2[1][0])).all()<=np.finfo(float).eps
                
        def test_sub_add_pow(self):
                # Create variables
                x, y = Node.create_nodes(2)
                # Define equation, variable values and seed
                f1 = x*y+3*x-2
                f2 = 2-x**2+2**y

        def test_logistic_arctan(self):
                # Create variables
                x, y = Node.create_nodes(2)
                # Define equation, variable values and seed
                f1 = arctan(x- y)
                f2 = logistic(x)
                values = [0, 0]
                seeds = [1, 1]
                # Evaluate
                res1 = Node.evaluate(f1, values, seeds) # single function
                res2 = Node.evaluate([f1, f2], values, seeds) #multi function
                assert (res1[0]==0)
                assert (res1[1]-(res2[1][0])).all()<=np.finfo(float).eps
                assert .5-(res2[0][1])<=np.finfo(float).eps
                

if __name__ == '__main__':
        unittest.main()
