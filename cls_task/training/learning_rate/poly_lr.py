

class PolyLR:
    def __init__(self, n_steps, offset_step=0, power=.9):
        self.n_steps = n_steps
        self.offset_step = offset_step
        self.power = power

    def step(self, step):
        return (1 - float(step - self.offset_step) / (self.n_steps - self.offset_step)) ** (self.power)
