from msagent import Worker, service, logger, event_handler


class Actor(Worker):
    def __init__(self, **kwargs):
        super().__init__()
        self.policy = 'dqn'
        self.eval_cnt = 1
        self.model = None

    @service()
    def policy_eval(self, msg):
        if self.model is not None:
            print("actor eval")
            self.eval_cnt += 1
            return "action"
        else:
            print("actor model not init")

    @event_handler("policy_update")
    def handler(self, msg):
        print("actor recv {}, update policy".format(msg))
        self.model = msg
