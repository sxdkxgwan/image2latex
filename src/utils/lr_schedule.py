import numpy as np


class LRSchedule(object):
    def __init__(self, lr_init, lr_min, start_decay=0, decay_rate=None, n_steps=None):
        # store parameters
        self.lr_init     = lr_init
        self.lr_min      = lr_min
        self.start_decay = start_decay
        self.decay_rate  = decay_rate # used when score improves
        self.n_steps     = n_steps # optional, if provided, decay is a 
                                   # smooth exp decay from init to min
        if self.n_steps is not None:
            self.exp_decay = np.power(lr_min/lr_init, 1/float(n_steps - start_decay))

        # initialize learning rate and score on eval
        self.score = None
        self.lr    = lr_init


    def update(self, t=None, score=None):
        """
        Update the learning rate:
            - decay by self.decay rate if score is lower than previous best
            - decay by self.decay_rate

        Both updates can concurrently happen if both
            - self.decay_rate is not None
            - self.n_steps is not None
        """
        if t is not None and t > self.start_decay and self.n_steps is not None:
            self.lr *= self.exp_decay

        if self.decay_rate is not None:
            if score is not None and self.score is not None:
                # assume lower is better
                if score > self.score:
                    self.lr *= self.decay_rate

        # update last score eval
        if score is not None:
            self.score = score

        self.lr = max(self.lr, self.lr_min)