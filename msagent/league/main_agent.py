import zmq
import numpy as np
import msagent.examples.ppo.learner.Learner as Player
from msagent.msagent.league.coordinator import pfsp


class MainAgent(Player):
    def __init__(self, agent, check_threshold):
        super(MainAgent, self).__init__()
        self.agent = agent
        self.checkpoint_step = 0
        self.gen = 1
        self.check_threshold = check_threshold
        self.model_key = "main_agent_{}".format(self.gen)
        self.context = zmq.Context()

    def pfsp_branch(self):
        historicals = [player for player in self.payoff.player]
        win_rates = self.payoff[self, historicals]
        return np.random.choice(historicals, p=pfsp(win_rates, weighting="squared"))

    def self_branch(self, opponent):
        if self.payoff[self, opponent] > 0.3:
            return opponent
        
        historicals = [player for player in self.payoff.players]
        win_rates = self.payoff[self, historicals]
        return np.random.choice(historicals, p=pfsp(win_rates, weighting="variance"))
    
    def create_checkpoint(self):
        self.pool_req_sock = self.context(zmq.REQ)
        self.pool_req_sock.connect("tcp://127.0.0.1:3421")
        self.pool_req_sock.send_string("write", zmq.SNDMORE)
        self.pool_req_sock.send_pyobj(self.agent)
        msg = self.pool_req_sock.recv_string()
        self.gen += 1

    def get_match(self):
        coin_toss = np.random.random()

        if coin_toss < 0.5:
            return self.pfsp_branch()
        
        main_agents =  [player for player in self.payoff.players if isinstance(player, MainAgent)]

        opponent = np.random.choice(main_agents)

        if coin_toss < 0.5 + 0.15:
            request = self.verfify_branch(opponent)
            return request
        
        return self.self_branch(opponent)