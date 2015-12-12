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

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1+T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''
        This function propagates the visible units activation puwards to the hidden untis

        Note that we return also the pre-sigmoid activation of the layer. As it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be needed to write down a more stable computational graph (see details in the reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units    '''
        # compute the activation of the hidden units given a sample of the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        #get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        ''' This function propagates the hidden units activation downwards to the visible units
        Note that we return also the pre_sigmoid_activation of the layer. As it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be needed to write down a more stable computational graph (see details in the reconstruction cost function)
        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hiddne units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        #get a sample of the visible given their activation
        # note that theano_rng.binomial returns a symbolic sample of dtype int64 by default. If
        # we want to keep our computations in floatX for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        '''
        This function implements one step of Gibbs sampling, starting from the hidden state
        '''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sample, starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        ''' This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable containing old state of Gibbs chain. This must be a shared vabiable of size (batch size, number of
        hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The dictionary contains the update rules for weights and bias but also an update of the shared variable used to store
        the persistent chain, if one is used.
        '''

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        #perform actual negative phase
        #in order to implement CD-k/PCD-k we need to scan over the function that implements one gibbs step k times.
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the 6th output
            outputs_info = [None, None, None, None, None, chain_start],
            n_steps = k
        )
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constan=[chain_end])

        #constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype = theano.config.floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])
        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        ''' Stochastic approximation to the pseudo-likelihood'''

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free nergy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:, bit_i_idx] = 1-xi[:, bit_i_idx], but assigns the result to xi_flip, instead of working in place on xi.
        xi_flip = T.setsubtensor(xi[:, bit_i_idx], 1-xi[:,bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        '''
        Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as input. To understand why this is so you need to understand a bit about how Theano works.
        Whenever you compile a Theano function, the cmoputaional graph that you pass as input gets optimized for speed and stability. This is done by changing several parts
        of the subgraphs with others. One such
        '''

