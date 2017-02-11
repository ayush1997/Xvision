#!/usr/bin/env python
class array(object):
    """Simple Array object that support autodiff."""

    def __init__(self, value, name=None):
        self.value = value
        if name:
            self.grad = lambda g: {name: g}

    def __add__(self, other):
        assert isinstance(other, int)
        ret = array(self.value + other)
        ret.grad = lambda g: self.grad(g)
        return ret

    def __mul__(self, other):
        assert isinstance(other, array)
        ret = array(self.value * other.value)

        def grad(g):
            x = self.grad(g * other.value)
            x.update(other.grad(g * self.value))
            return x

        ret.grad = grad
        return ret


# some examples
a = array(1, 'a')
b = array(2, 'b')
c = b * a
d = c + 1
print d.value
print d.grad(1)


import numpy as np
a = np.ones(10)
b = np.ones(10) * 2
c = b * a
d = c + 1

import mxnet as mx
kv = mx.kv.create('local') # create a local kv store.
shape = (2,3)
kv.init(7, mx.nd.ones(shape)*2)
a = mx.nd.zeros(shape)
kv.pull(7, out = a)
print a.asnumpy()


import time
import sys
def testmain():
    print("Test1 args are" + str( sys.argv))

if __name__ == '__main__':
    testmain()
