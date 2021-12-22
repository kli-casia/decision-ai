from msagent import Worker, service, logger, EventDispatcher
from model import ActorCritic

class Learner(Worker):

    ed = EventDispatcher()

    def __init__(self,):
        super().__init__()
        self.model = None

    @service()
    def make_model(self, num_inputs, num_actions):
        if self.model == None:
            self.model = ActorCritic(num_inputs, num_actions)
            self.ed.fire('policy_update', payload=self.model)

    @service(batch=3)
    def policy_train(self, msg, clip=0.2, lr=3e-4):
        logger.info('policy train')
        logger.info(clip)
        result = self.model.train(msg, clip, lr)
        self.ed.fire('policy_update', payload=self.model)
        return [result for i in range(len(msg))]