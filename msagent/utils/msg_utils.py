import pickle5 as pickle


class ServiceMsg():
    def __init__(self, func_name, args, kwargs):
        self._entry_name = func_name
        self._data = {
            'args': args,
            'kwargs': kwargs,
        }

    @property
    def func_name(self,):
        return self._entry_name

    @property
    def args(self,):
        return self._data['args']

    @property
    def kwargs(self,):
        return self._data['kwargs']


class EventMsg():
    def __init__(self, event_name, payload):
        self._event_name = event_name
        self._data = payload

    @property
    def event_name(self,):
        return self._event_name

    @property
    def payload(self,):
        return self._data


class RegisterMsg():
    def __init__(self, entry_name, service_name, ident, tags=[]):
        self._entry_name = entry_name
        self._service_name = service_name
        self._service_ident = ident
        self._tags = []
        if tags:
            assert isinstance(tags, list)
            self._tags = tags

    @property
    def entry_name(self,):
        return self._entry_name

    @property
    def service_name(self,):
        return self._service_name

    @property
    def tags(self,):
        return self._tags

    @property
    def ident(self,):
        return self._service_ident


class DiscoverMsg():
    def __init__(self, entry_name):
        self._entry_name = entry_name

    @property
    def entry_name(self,):
        return self._entry_name


def SerializeMsg(msg):
    return pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)


def DeSerializeMsg(msg):
    return pickle.loads(msg)
