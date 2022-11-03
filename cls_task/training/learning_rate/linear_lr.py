

class LinearLR:
    def __init__(self, n_steps, offset_step=0):
        self.n_steps = n_steps
        self.offset_step = offset_step

    def step(self, step):
        return (1 - float(step - self.offset_step) / (self.n_steps - self.offset_step))