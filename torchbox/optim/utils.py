
def debias(mom, damp, step):
    return damp * (1 - mom**step) / (1-mom)
