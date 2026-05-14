from collections import deque
import random
class ReplayMemory():
    def __init__(self,maxlen,seed=None):
        self.memory = deque([],maxlen=maxlen)
        if seed is not None:
            random.seed(seed)
    def append(self,transition):
        self.memory.append(transition)

    def sample(self,batch_size):
        #constrain the batch_size to the max size of the memory
        return random.sample(self.memory,batch_size if batch_size <= len(self.memory) else len(self.memory))
    
    def __len__(self):
        return len(self.memory)
