from msagent.core.register import ConsulClient
from msagent.utils import service_utils
from msagent.utils import msg_utils
from msagent import config as conf
from msagent.utils import logger

import pyarrow.plasma as plasma
from collections import deque
import zmq


def route_event_msg(services):
    # router meta
    _services = services
    _host = service_utils.GetLocalIP()
    _port = conf.PUB_PORT
    _en2service = {}
    _en2tags = {}

    _worker2service = {}
    _en2last_event_bin_msg_id = {}
    _service2ens = {s: [] for s in _services}
    _service2all_workers = {s: [] for s in _services}
    _service2ready_workers = {s: [] for s in _services}

    ctx = zmq.Context.instance()
    _frontend = ctx.socket(zmq.ROUTER)
    _frontend.bind("tcp://*:{}".format(_port))
    _frontend.set_hwm(conf.ZMQ_HWM)

    _backend = ctx.socket(zmq.ROUTER)
    _backend.bind(service_utils.GetEventBackendAddr(bind=True))

    _ctrlend = ctx.socket(zmq.REP)
    _ctrlend.bind(service_utils.GetEventCtrlendAddr(bind=True))

    if conf.USE_PLASMA:
        _plasma_client = plasma.connect(service_utils.GetPlasmaDir())

    if conf.IS_DISTRIBUTED:
        _consul_client = ConsulClient(conf.CONSUL_PORT)

    _poller = zmq.Poller()
    _poller.register(_backend, zmq.POLLIN)
    _poller.register(_ctrlend, zmq.POLLIN)
    _poller.register(_frontend, zmq.POLLIN)

    def _all_worker_is_ready():
        for s in services:
            if not len(_service2ready_workers[s]) == len(_service2all_workers[s]):
                logger.debug(
                    "not all workers is ready: {} vs. {}".format(
                        _service2ready_workers, _service2all_workers
                    )
                )
                return False
        return True

    # start proxy loop
    logger.info("event: start event proxy loop ...")
    backend_ready = True
    while True:
        socks = dict(_poller.poll())
        if _frontend in socks:
            assert _all_worker_is_ready()
            client, bin_entry_name, msg = _frontend.recv_multipart()
            logger.debug("event recv {} msg".format(bin_entry_name))

            if conf.IS_DISTRIBUTED and conf.USE_PLASMA:
                # if is distributed, the recv msg is serialized
                # we just need to put it in plasma
                msg = _plasma_client.put(msg).binary()
            else:
                # if local, the recv msg is just the binary msg_id
                # if not use plasma, the recv msg is the serialized data
                msg = msg

            en = bin_entry_name.decode("ascii")
            if en not in _en2service or en not in _en2tags:
                logger.error("entryname {} not found".format(en))
            service_name = _en2service[en]
            tags = _en2tags[en]

            all_workers = _service2all_workers[service_name]
            for worker in all_workers:
                _backend.send_multipart([worker, client, bin_entry_name, msg])
            logger.debug("broadcast {} msg to workers: {}".format(en, all_workers))

            _service2ready_workers[service_name] = []

            if conf.USE_PLASMA:
                # delete last event msg if use plasma
                last_event_bin_msg_id = _en2last_event_bin_msg_id.get(en)
                if last_event_bin_msg_id is not None:
                    last_msg_id = plasma.ObjectID(last_event_bin_msg_id)
                    logger.debug("event: delete {} last msg".format(en))
                    _plasma_client.delete([last_msg_id])
            # update last event msg
            _en2last_event_bin_msg_id[en] = msg

            if not _all_worker_is_ready() and backend_ready:
                logger.debug("not all workers is ready, stop poll in events")
                _poller.unregister(_frontend)
                backend_ready = False

        if _backend in socks:
            worker, typed = _backend.recv_multipart()
            if typed == b"READY":
                logger.debug("worker {} is ready".format(worker))
                assert worker in _worker2service
                service_name = _worker2service[worker]
                _service2ready_workers[service_name].append(worker)

                if _all_worker_is_ready() and not backend_ready:
                    logger.debug("router start poll in events")
                    _poller.register(_frontend, zmq.POLLIN)
                    backend_ready = True
            else:
                logger.error("unknown event backend msg type")

        if _ctrlend in socks:
            serial_msg = _ctrlend.recv()
            msg = msg_utils.DeSerializeMsg(serial_msg)

            if isinstance(msg, msg_utils.RegisterMsg):
                service_name = msg.service_name
                entry_name = msg.entry_name
                tags = msg.tags
                ident = msg.ident
                logger.debug("recv register msg for {}".format(entry_name))
                assert service_name in _services

                if ident not in _worker2service:
                    # the worker is not registered before
                    logger.debug("{} register event entry {}".format(ident, entry_name))
                    _worker2service[ident] = service_name
                    _service2all_workers[service_name].append(ident)
                    _service2ready_workers[service_name].append(ident)

                if entry_name in _en2last_event_bin_msg_id:
                    # the worker missed the cached last event
                    # so resend last event to the worker
                    logger.debug(
                        "worker {} missed event {}, resend last msg ...".format(
                            ident, entry_name
                        )
                    )
                    bin_entry_name = entry_name.encode("ascii")
                    if conf.USE_PLASMA:
                        # resend serial msg instead of msg id
                        # since the slow worker should manage the msg lifetime
                        bin_msg_id = _en2last_event_bin_msg_id[entry_name]
                        msg_id = plasma.ObjectID(bin_msg_id)
                        serial_msg = _plasma_client.get(msg_id)
                    else:
                        # if not use plasma, the bin msg id is actually serial msg
                        serial_msg = _en2last_event_bin_msg_id[entry_name]
                    _backend.send_multipart(
                        [ident, b"RESEND", bin_entry_name, serial_msg]
                    )

                if entry_name in _en2service:
                    # check if the entrypoint is registered
                    # if it is, do nothing
                    assert _en2service[entry_name] == service_name
                    logger.debug("entrypoint {} has registered".format(entry_name))
                    _ctrlend.send(b"ok")
                else:
                    if conf.IS_DISTRIBUTED:
                        # register to consul if in distributed mode
                        ok = _consul_client.register_service(
                            entry_name, _host, _port, tags=tags
                        )
                        if not ok:
                            logger.error("register entrypoint fail")
                            raise RuntimeError("register entrypoint fail")
                        logger.debug(
                            "register entrypoint {} to consul".format(entry_name)
                        )

                    _service2ens[service_name].append(entry_name)
                    _en2service[entry_name] = service_name
                    _en2tags[entry_name] = tags
                    _ctrlend.send(b"ok")

            elif isinstance(msg, msg_utils.DiscoverMsg):
                if conf.IS_DISTRIBUTED:
                    logger.error(
                        "can not get service from router if in distributed mode"
                    )
                    raise RuntimeError(
                        "can not get service from router if in distributed mode"
                    )

                entry_name = msg.entry_name
                logger.debug("get event handler for entry: {}".format(entry_name))
                if entry_name in _en2service and entry_name in _en2tags:
                    tags = _en2tags[entry_name]
                    addr = "tcp://{}:{}".format(_host, _port)
                    _ctrlend.send(msg_utils.SerializeMsg([addr, tags]))
                else:
                    logger.warning("discover no service for : {}".format(entry_name))
                    _ctrlend.send(msg_utils.SerializeMsg([None, None]))
            else:
                raise NotImplementedError("unknown type of ctrl msg")


def route_service_msg(services, max_queue_len=10):
    # router meta
    _services = services
    _host = service_utils.GetLocalIP()
    _port = conf.ROUTER_PORT
    _en2service = {}
    _en2tags = {}

    _worker2service = {}
    _service2queue = {s: deque(maxlen=max_queue_len) for s in _services}
    _service2ready_workers = {s: [] for s in _services}

    # start local zmq sockets
    ctx = zmq.Context.instance()
    _frontend = ctx.socket(zmq.ROUTER)
    _frontend.bind("tcp://*:{}".format(_port))
    _frontend.set_hwm(conf.ZMQ_HWM)

    _backend = ctx.socket(zmq.ROUTER)
    _backend.bind(service_utils.GetServiceBackendAddr(bind=True))

    _ctrlend = ctx.socket(zmq.REP)
    _ctrlend.bind(service_utils.GetServiceCtrlendAddr(bind=True))

    _poller = zmq.Poller()
    _poller.register(_backend, zmq.POLLIN)
    _poller.register(_ctrlend, zmq.POLLIN)

    _counter = service_utils.QPSCounter(name="router")

    if conf.IS_DISTRIBUTED:
        _consul_client = ConsulClient(conf.CONSUL_PORT)

    if conf.USE_PLASMA:
        _plasma_client = plasma.connect(service_utils.GetPlasmaDir())

    def _no_service_queue_is_full():
        for service, queue in _service2queue.items():
            assert len(queue) <= queue.maxlen
            if len(queue) == queue.maxlen:
                logger.info(
                    "service queue {} is full: {}".format(service, _service2queue)
                )
                return False
        return True

    # start proxy loop
    logger.info("service: start service proxy loop ...")
    _counter.start()
    backend_ready = False
    while True:
        sockets = dict(_poller.poll())

        if _backend in sockets:
            data = _backend.recv_multipart()
            worker, client = data[:2]
            if client == b"READY" and len(data) == 2:
                # if client is b'READY', the msg is a 'ready msg'
                # the 'ready msg' will tell router which worker is ready for next msg
                logger.debug("worker {} ready".format(worker))
                assert worker in _worker2service
                service = _worker2service[worker]
                # if the queue is not empty, dispatch one task to the worker instantly
                # otherwise mark it as 'ready'
                queue = _service2queue[service]
                if len(queue) > 0:
                    client, bin_entry_name, msg = queue.popleft()
                    # the msg is msg_id if use plasma otherwise serialized data
                    _backend.send_multipart([worker, client, bin_entry_name, msg])
                else:
                    _service2ready_workers[service].append(worker)
            else:
                # if not 'ready' msg, send out the msg to frontend
                assert len(data) == 3
                worker, client, msg = data
                logger.debug("service: get msg from backend: {}".format(worker))
                _frontend.send_multipart([client, msg])

            if _no_service_queue_is_full() and not backend_ready:
                logger.debug("service: start pollin")
                _poller.register(_frontend, zmq.POLLIN)
                backend_ready = True

        if _frontend in sockets:
            client, bin_entry_name, msg = _frontend.recv_multipart()
            logger.debug("frontend recv {} msg from {}".format(bin_entry_name, client))
            _counter.count(msg)

            if conf.IS_DISTRIBUTED and conf.USE_PLASMA:
                # if is distributed, the recv msg is serialized
                # we just need to put it in plasma
                msg = _plasma_client.put(msg).binary()
            else:
                # if local, the recv msg is just the binary msg_id
                # if not use plasma, the recv msg is the serialized data
                msg = msg

            en = bin_entry_name.decode("ascii")
            if en not in _en2service or en not in _en2tags:
                raise RuntimeError("service: entryname {} not found".format(en))

            service_name = _en2service[en]
            assert service_name in _service2ready_workers

            workers = _service2ready_workers[service_name]
            if len(workers) > 0:
                # if service has ready worker, dispatch the msg to one
                worker = workers.pop(0)
                _backend.send_multipart([worker, client, bin_entry_name, msg])
                logger.debug("frontend send {} msg to worker {}".format(en, worker))
            else:
                # otherwise add the msg to the queue
                logger.debug("frontend add {} msg to {} queue".format(en, service))
                queue = _service2queue[service_name]
                queue.append([client, bin_entry_name, msg])

            if not _no_service_queue_is_full() and backend_ready:
                # frontend will stop pollin msg under any queue of service is full
                logger.debug("some service queue is full, stop pollin")
                _poller.unregister(_frontend)
                backend_ready = False

        if _ctrlend in sockets:
            # this part is almost the same as router event msg
            serial_msg = _ctrlend.recv()
            msg = msg_utils.DeSerializeMsg(serial_msg)

            if isinstance(msg, msg_utils.RegisterMsg):
                service_name = msg.service_name
                entry_name = msg.entry_name
                tags = msg.tags
                ident = msg.ident
                logger.debug("service: recv register msg for {}".format(entry_name))
                assert service_name in _services
                # record worker ident
                _worker2service[ident] = service_name

                if entry_name in _en2service:
                    # check if the entrypoint is registered
                    assert _en2service[entry_name] == service_name
                    logger.debug(
                        "service: entrypoint {} has registered".format(entry_name)
                    )
                    _ctrlend.send(b"ok")
                else:
                    if conf.IS_DISTRIBUTED:
                        # register to consul if in distributed mode
                        ok = _consul_client.register_service(
                            entry_name, _host, _port, tags=tags
                        )
                        if not ok:
                            logger.error("service: register entrypoint fail")
                        logger.debug(
                            "service: register entrypoint {} to consul".format(
                                entry_name
                            )
                        )

                    _en2service[entry_name] = service_name
                    _en2tags[entry_name] = tags
                    _ctrlend.send(b"ok")

            elif isinstance(msg, msg_utils.DiscoverMsg):
                if conf.IS_DISTRIBUTED:
                    raise RuntimeError(
                        "can not get service from router if in distributed mode"
                    )

                entry_name = msg.entry_name
                logger.debug("service: get service for : {}".format(entry_name))
                if entry_name in _en2service and entry_name in _en2tags:
                    tags = _en2tags[entry_name]
                    addr = "tcp://{}:{}".format(_host, _port)
                    _ctrlend.send(msg_utils.SerializeMsg([addr, tags]))
                else:
                    logger.warning("service: no service for : {}".format(entry_name))
                    _ctrlend.send(msg_utils.SerializeMsg([None, None]))
            else:
                raise NotImplementedError("unknown type of ctrl msg")
