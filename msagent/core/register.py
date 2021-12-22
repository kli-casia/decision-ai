from msagent.utils import logger
from consul import Consul, Check
from random import choices


class ConsulClient:
    def __init__(self, port=8500):
        self.consul = Consul(port=port)

    def register_service(self, service_name, host, port, tags=None, interval=None):
        tags = tags or []
        service_id = "{}_{}".format(service_name, host)
        check = Check().tcp(host, port, "5s", "30s", "30s")
        ok = self.consul.agent.service.register(
            service_name,
            service_id=service_id,
            address=host,
            port=port,
            tags=tags,
            interval=interval,
            check=check,
        )
        if not ok:
            logger.error("register fail")
        return ok

    def deregister_service(self, service_id):
        ok = self.consul.agent.service.deregister(service_id)
        if not ok:
            logger.error("deregister fail")
        return ok

    def get_service(self, service_name, get_all):
        # get services after health checking
        _, nodes = self.consul.health.service(service=service_name, passing=True)
        assert nodes, "service not found"

        addrs = []
        tags = []
        for node in nodes:
            service = node.get("Service")
            addr = "tcp://{}:{}".format(service["Address"], service["Port"])
            if service["Tags"]:
                if tags:
                    assert tags == service["Tags"], "tags must be the same"
                else:
                    tags = service["Tags"]
            addrs.append(addr)

        if not get_all:
            addr = choices(addrs, k=1)[0]
            return addr, tags

        return addrs, tags
