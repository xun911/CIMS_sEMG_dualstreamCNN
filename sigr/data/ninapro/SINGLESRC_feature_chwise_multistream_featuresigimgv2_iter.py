from __future__ import division
import os
from nose.tools import assert_equal
from functools import partial
from logbook import Logger
import mxnet as mx
import numpy as np
import scipy.io as sio
import scipy.stats as sstat
from itertools import product
from collections import OrderedDict
#from lru import LRU
from . import Dataset as Base
from ... import emg_features
from .. import Combo, Trial
from ... import utils, constant
import types
from ...genIndex import genIndex



class ChwiseFeature_FeatureSigImgv2_Iter(mx.io.NDArrayIter):

    def __init__(self, *args, **kargs):
        print 'Initialization Data Iter!'
        self.random_shift_vertical = kargs.pop('random_shift_vertical', 0)
        self.random_shift_horizontal = kargs.pop('random_shift_horizontal', 0)
        self.random_shift_fill = kargs.pop('random_shift_fill', constant.RANDOM_SHIFT_FILL)
        self.amplitude_weighting = kargs.pop('amplitude_weighting', False)
        self.amplitude_weighting_sort = kargs.pop('amplitude_weighting_sort', False)
        self.downsample = kargs.pop('downsample', None)
        self.shuffle = kargs.pop('shuffle', False)
        self.adabn = kargs.pop('adabn', False)
        self._gesture = kargs.pop('gesture')
        self._subject = kargs.pop('subject')
        self._segment = kargs.pop('segment')
        self._index_orig = kargs.pop('index')
        self._index = np.copy(self._index_orig)
#        self.feature_list = kargs.pop('feature_list')
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject')
        self.mini_batch_size = kargs.pop('mini_batch_size', kargs.get('batch_size'))
        self.random_state = kargs.pop('random_state', np.random)
        self.balance_gesture = kargs.pop('balance_gesture', 0)  
        self.window = kargs.pop('window', 0)
        self.num_semg_col = kargs.pop('num_semg_col')
        self.num_semg_row = kargs.pop('num_semg_row')

       
        
        super(ChwiseFeature_FeatureSigImgv2_Iter, self).__init__(*args, **kargs)
        
                  
        
  
        self.num_data = len(self._index)
        self.data_orig = self.data
        self.reset()
                
        
        self.window = 1
        
        
        
                                                                   
#        self.shape1 = self.data[0][1].shape[2]
#        self.shape2 = self.data[1][1].shape[2]
#        print ("data shape 1 = (%d, %d), data shape 2 = (%d, %d)" % (self.data[0][1].shape[2], 
#                                                                     self.data[0][1].shape[3],
#                                                                     self.data[1][1].shape[2],
#                                                                     self.data[1][1].shape[3]))
        
        
        self.num_channel = []
        self.shapes1 = []
        self.shapes2 = []        
        
        self.num_channel.append(self.data[0][1].shape[2])
        self.shapes1.append(len(genIndex(self.num_semg_col*self.num_semg_row))-1)
        self.shapes2.append(1)          
        
        
        total_ch = self.num_semg_col*self.num_semg_row        
            
        for i in range(total_ch):
            self.num_channel.append(1)
            self.shapes1.append(1)
            self.shapes2.append(32)
            
        for i in range(total_ch):
            self.num_channel.append(1)
            self.shapes1.append(1)
            self.shapes2.append(22)     
        
#        print self.shapes1
#        print self.shapes2
        

    @property
    def num_sample(self):
        return self.num_data

    @property
    def gesture(self):
        return self._gesture[self._index]

    @property
    def subject(self):
        return self._subject[self._index]

    @property
    def segment(self):
        return self._segment[self._index]

    @property
    def provide_data(self):

#         assert_equal(len(self.data), 1)
         res = [('stream%d_data' % i, tuple([self.batch_size, ch] + list([self.shapes1[i], self.shapes2[i]])))
                   for i, ch in enumerate(self.num_channel)]         
         
         return res
         
             

    def _expand_index(self, index):
        return np.hstack([np.arange(i, i + self.window) for i in index])

    def _reshape_data(self, data):
        return data.reshape(-1, self.window, *data.shape[2:])
        
    def _get_sigimg(self, data):
        
        from ... import Context
        import joblib as jb
        res = []
         
        for amp in Context.parallel(jb.delayed(_get_sigimg_aux)(sample) for sample in data):
            res.append(amp[np.newaxis, ...])
        res = np.concatenate(res, axis=0)    
#        res = res.reshape(res.shape[0], 1, res.shape[1], res.shape[2])
        res = res.reshape(res.shape[0], res.shape[1], res.shape[2], -1)
        return res

    def _get_ch(self, data):

#        print data.shape
        res = []
        data_1 = data[:,0:32,:]
        data_2 = data[:,32:54,:]
#        print data_1.shape
#        print data_2.shape
        for i in range(data_1.shape[2]):
              ch_data = data_1[:,:,i]
              ch_data = ch_data.reshape(ch_data.shape[0], 1, -1, ch_data.shape[1])
              res.append(ch_data)    
              
        for i in range(data_2.shape[2]):
              ch_data = data_2[:,:,i]
              ch_data = ch_data.reshape(ch_data.shape[0], 1, -1, ch_data.shape[1])
              res.append(ch_data)       
              
              
        return res      

    def _get_segments(self, a, index):
        b = mx.nd.empty((len(index), self.window) + a.shape[2:], dtype=a.dtype)  
        for i, j in enumerate(index):           
            b[i] = a[j:j + self.window].reshape(self.window, *a.shape[2:])
        return b

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset."
        
        assert self.window == 1    
                  
        if data_source is self.data and self.window > 1:  
            if self.cursor + self.batch_size <= self.num_data:
                #  res = [self._reshape_data(x[1][self._expand_index(self._index[self.cursor:self.cursor+self.batch_size])]) for x in data_source]
                res = [self._get_segments(x[1], self._index[self.cursor:self.cursor+self.batch_size]) for x in data_source]
            else:
                pad = self.batch_size - self.num_data + self.cursor
                res = [(np.concatenate((self._reshape_data(x[1][self._expand_index(self._index[self.cursor:])]),
                                        self._reshape_data(x[1][self._expand_index(self._index[:pad])])), axis=0)) for x in data_source]
        else:
            if self.cursor + self.batch_size <= self.num_data:
                res = [(x[1][self._index[self.cursor:self.cursor+self.batch_size]]) for x in data_source]
            else:
                pad = self.batch_size - self.num_data + self.cursor
                res = [(np.concatenate((x[1][self._index[self.cursor:]], x[1][self._index[:pad]]), axis=0)) for x in data_source]       
        
        if data_source is self.data:       
             new_res = []           
             for a in res:
                 new_res.append(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a)
             res = new_res 
#             print res[0].shape     
             for a in res:
                 assert(a.shape[1] == 1)
                    
             res = [a.reshape(a.shape[0], a.shape[2], a.shape[3]) for a in res]             
#             assert (len(res) == 1)   
#             kk = res[0]
#             assert (len(res) == 2)
             res_ = [] 
#             print res[0].shape
             res_.append(self._get_sigimg(res[0].asnumpy() if isinstance(res[0], mx.nd.NDArray) else res[0]))
             
                          
#             print res_[0].shape
             
             res_ = res_ + self._get_ch(res[0])             
             res = res_
               
#             print len(res)
#             print len(self.num_channel)                      
             assert (len(res) == len(self.num_channel))

        res = [a if isinstance(a, mx.nd.NDArray) else mx.nd.array(a) for a in res]
        return res

    def _rand(self, smin, smax, shape):
        return (smax - smin) * self.random_state.rand(*shape) + smin

    def _do_shuffle(self):
        if not self.adabn or len(set(self._subject)) == 1:
            self.random_state.shuffle(self._index)
        else:
            batch_size = self.mini_batch_size
            # batch_size = self.batch_size
            # logger.info('AdaBN shuffle with a mini batch size of {}', batch_size)
            self.random_state.shuffle(self._index)
            subject_shuffled = self._subject[self._index]
            index_batch = []
            for i in sorted(set(self._subject)):
                index = self._index[subject_shuffled == i]
                index = index[:len(index) // batch_size * batch_size]
                index_batch.append(index.reshape(-1, batch_size))
            index_batch = np.vstack(index_batch)
            index = np.arange(len(index_batch))
            self.random_state.shuffle(index)
            self._index = index_batch[index, :].ravel()

            for i in range(0, len(self._subject), batch_size):
                # Make sure that the samples in one batch are from the same subject
                assert np.all(self._subject[self._index[i:i + batch_size - 1]] ==
                              self._subject[self._index[i + 1:i + batch_size]])

            if batch_size != self.batch_size:
                assert self.batch_size % batch_size == 0
                assert (self.batch_size // batch_size) % self.num_subject == 0
                self._index = self._index[:len(self._index) // self.batch_size * self.batch_size].reshape(
                    -1, self.num_subject, batch_size).transpose(0, 2, 1).ravel()

    def reset(self):
        self._reset()
        super(ChwiseFeature_FeatureSigImgv2_Iter, self).reset()

    def _reset(self):
        self._index = np.copy(self._index_orig)

#        if self.amplitude_weighting:
#            assert np.all(self._index[:-1] < self._index[1:])
#            if not hasattr(self, 'amplitude_weight'):
#                self.amplitude_weight = get_amplitude_weight(
#                    self.data[0][1], self._segment, self.framerate)
#            if self.shuffle:
#                random_state = self.random_state
#            else:
#                random_state = np.random.RandomState(677)
#            self._index = random_state.choice(
#                self._index, len(self._index), p=self.amplitude_weight)
#            if self.amplitude_weighting_sort:
#                logger.debug('Amplitude weighting sort')
#                self._index.sort()

        if self.downsample:
            samples = np.arange(len(self._index))
            np.random.RandomState(667).shuffle(samples)
            assert self.downsample > 0 and self.downsample <= 1
            samples = samples[:int(np.round(len(samples) * self.downsample))]
            assert len(samples) > 0
            self._index = self._index[samples]

        if self.balance_gesture:
            num_sample_per_gesture = int(np.round(self.balance_gesture *
                                                  len(self._index) / self.num_gesture))
            choice = []
            for gesture in set(self.gesture):
                mask = self._gesture[self._index] == gesture
                choice.append(self.random_state.choice(np.where(mask)[0],
                                                       num_sample_per_gesture))
            choice = np.hstack(choice)
            self._index = self._index[choice]

        if self.shuffle:
            self._do_shuffle()

        self.num_data = len(self._index)


def _get_sigimg_aux(data):               
    return np.transpose(get_sig_img(data.T)) 

def get_sig_img(data):
     ch_num = data.shape[0]
     index = genIndex(ch_num)
     signal_img = data[index]
     signal_img = signal_img[:-1]
#     print signal_img.shape
     return signal_img  