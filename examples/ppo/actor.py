from msagent import Worker, service, logger, event_handler


class Actor(Worker):
    def __init__(self, policy="ppo", **kwargs):
        super().__init__()
        self.policy = policy
        self.eval_cnt = 1
        self.model = None

    @service
    def policy_eval(self, msg):
        if self.model is not None:
            logger.info("actor get: {}".format("state"))
            logger.info("{} eval".format(self.policy))
            return self.model(msg)
        else:
            print("model not init")

    @event_handler("policy_update")
    def handler(self, msg):
        self.model = msg
        logger.info("recv broadcast, update policy")
