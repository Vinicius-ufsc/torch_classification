from metrics import ComputeMetrics

class Callbacks():

    def __init__(self):
        self._callbacks = {
            'on_train_batch_start' : [],
        }

    def run(self, hook):
        for task in self._callbacks[hook]:
            pass