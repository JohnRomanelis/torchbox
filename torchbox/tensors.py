import torch


class LossTensor:
    """ 
        A torch.Tensor with a dict!

        This class can be used when the total loss is composed of different loss.
        We can store these losses in the sublosses dict to access them by the recorder.
    """
    def __init__(self, data, losses={}, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self._losses = losses
    def __repr__(self):
        return "Sublosses:\n{}\n\ndata:\n{}".format(self._losses, self._t)
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs={}
        args = [a._t if hasattr(a, '_t') else a for a in args]
        ret = func(*args, **kwargs)
        return LossTensor(ret, losses=self._losses)
    
    # access to tensor opperations like sum, add, backward etc
    # without this line we can call torch.add(a, b), of a.add(b) 
    # where a is a tensor and b a LossTensor
    # but we cannot call b.add(a), or b.backward()
    def __getattr__(self, k): return getattr(self._t, k)
    
    
    # Dictionary related operations
    def add_sublosses(self, sublosses_dict):
        assert isinstance(sublosses_dict, dict)
        for k, v in sublosses_dict.items():
            self.add_subloss(k, v)
        
    def add_subloss(self, key, value):
        if isinstance(value, torch.Tensor):
            self._losses[key] = value.item()
        else:
            self._losses[key] = value
    
    @property
    def sublosses(self):
        return self._losses
    
    @sublosses.setter
    def sublosses(self, losses_dict):
        assert isinstance(losses_dict, dict)
        self._losses = losses_dict
    


''' # Deprecated! Doesn't work if data are already in the GPU
class LossTensor(torch.Tensor):
    """ 
        A torch.Tensor with a dict!

        This class can be used when the total loss is composed of different loss.
        We can store these losses in the sublosses dict to access them by the recorder.
    """
    def __init__(self, x, *args, **kwargs):
        self._sublosses = {}
    
    def add_sublosses(self, sublosses_dict):
        assert isinstance(sublosses_dict, dict)
        for k, v in sublosses_dict.items():
            self.add_subloss(k, v)
        
    def add_subloss(self, key, value):
        if isinstance(value, torch.Tensor):
            self._sublosses[key] = value.item()
        else:
            self._sublosses[key] = value
            
    def __repr__(self):
        ret = super().__repr__()
        if len(list(self._sublosses)) > 0:
            ret = ret + '\n' + self._sublosses.__repr__()
        return ret
    
    @property
    def sublosses(self):
        return self._sublosses
    
    @sublosses.setter
    def sublosses(self, sublosses_dict):
        assert isinstance(sublosses_dict, dict)
        self._sublosses = sublosses_dict
'''

