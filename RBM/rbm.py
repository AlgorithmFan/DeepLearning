#!usr/bin/env python
#coding:utf-8

import numpy as np
import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams



class RBM(object):
    '''Restricted Boltzmann Machine'''
    def __init__(self, input=None,
                 n_visible=784,
                 n_hidden=500,
                 W=None,
                 hbias=None,
                 vbias=None,
                 numpy_rng=None,
                 theano_rng=None):
        '''
        RBM constructor. Defines the parameters of the model along with the basic operations for inferring hidden from visible
        as well as for performing CD updates.

        :param input:
        :param n_visible: number of visible units
        :param n_hidden: number of hidden units
        :param W: the weights between the visible units and hidden units
        :param hbias:
        :param vbias:
        '''
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if W is None:
            initial_W = np.asarray(
                numpy_rng.uniform(
                low = -4*np.sqrt(6. / (n_hidden+n_visible)),
                high = 4*np.sqrt(6. / (n_hidden+n_visible)),
                size = (n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            hbias = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name = 'hbias',
                borrow=True
            )

        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(n_visible, dtype=theano.config.floatX),
                name = 'vbias',
                borrow = True
            )

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]
