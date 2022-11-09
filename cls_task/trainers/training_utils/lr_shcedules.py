

class LinearLR:
    def __init__(self, n_steps, offset_step=0):
        self.n_steps = n_steps
        self.offset_step = offset_step

    def step(self, step):
        return (1 - float(step - self.offset_step) / (self.n_steps - self.offset_step))
    



class PolyLR:
    def __init__(self, n_steps, offset_step=0, power=.9):
        self.n_steps = n_steps
        self.offset_step = offset_step
        self.power = power

    def step(self, step):
        return (1 - float(step - self.offset_step) / (self.n_steps - self.offset_step)) ** (self.power)


def get_lr_schedule(name, ):

    if name == "linear":
        return LinearLR
    elif name == "poly":
        return PolyLR
    else:
        raise Exception("learning rate schedule name {} dosen't implemented!!".format(name))