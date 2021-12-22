import os

_default_config = {
    "IS_DISTRIBUTED": True,
    "USE_PLASMA": False,
    "PLASMA_CAPACITY": 5,
    "PUB_PORT": 9900,
    "ROUTER_PORT": 9800,
    "CONSUL_PORT": 8500,
    "SERVICE_TIMEOUT": 30,
    "DEFAULT_TMP_DIR": ".msagent",
    "LOG_DIR": "log",
    "LOG_LEVEL": "debug",
    "ZMQ_HWM": 10,
}


class Config:
    def __init__(self):
        self.__dict__.update(_default_config)
        if os.getenv("msagent_CONFIG") is not None:
            _user_config = os.getenv("msagent_CONFIG")
            self.update(eval(_user_config))

    def update(self, conf):
        assert isinstance(conf, dict)
        for key, value in conf.items():
            assert isinstance(key, str)
            key = key.upper()
            self.__dict__.update({key: value})


config = Config()
