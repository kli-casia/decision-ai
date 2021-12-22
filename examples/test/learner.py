from msagent import Worker, service, logger, EventDispatcher, config
import numpy as np

class Learner(Worker):

    def __init__(self, **kwargs):
        super().__init__()
        self.train_cnt = 0
        self.model = None
        self.ed = EventDispatcher()

    @service
    def make_model(self, msg):
        if not self.model:
            print('learner recv {}, make model'.format(msg))
            self.model = 'model'
            self.ed.fire('policy_update', payload=self.model)
        else:
            print('model already init')

    @service(response=False)
    def policy_train(self, msg):
        print('learner trained')
        self.train_cnt += 1
        model = np.ones([500, 500], dtype=np.float32)
        self.model = 'model_{}'.format(self.train_cnt)
        self.ed.fire('policy_update', payload=model)
