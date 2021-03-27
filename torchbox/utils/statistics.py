import torch 


def get_actual_elems(sp, mask=None):
    if mask is not None:
        return sp[mask]
    
    mask = torch.zeros_like(sp)
    mask[sp!=0] = 1
    mask = mask.type(torch.bool)
    return sp[mask]

def sparse_mean(sp, mask=None):
    elems = get_actual_elems(sp, mask)
    return elems.mean()

def sparse_std(sp, mask=None):
    elems = get_actual_elems(sp, mask)
    return elems.std()    

def sparse_statistics(sp, mask=None):
    elems = get_actual_elems(sp)
    mean = elems.mean()
    std  = elems.std()
    maxv = elems.max()
    minv = elems.min()
    
    print('''
    mean : {0:0.4f}
    std  : {1:0.4f}
    max  : {2:0.4f}
    min  : {3:0.4f}
    '''.format(mean.item(), std, maxv, minv))


def dense_statistics(elems):
    mean = elems.mean()
    std  = elems.std()
    maxv = elems.max()
    minv = elems.min()
    
    print('''
    mean : {0:0.4f}
    std  : {1:0.4f}
    max  : {2:0.4f}
    min  : {3:0.4f}
    '''.format(mean, std, maxv, minv)) 
