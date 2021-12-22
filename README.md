# msagent

MSAgent 是一套基于微服务架构的分布式强化学习框架
## Get Started



1. 安装consul

官方安装地址: https://www.consul.io/downloads


2. 安装Redis

官方安装网址：https://redis.io/download


3. 克隆项目代码

```
git clone git@github.com:CodeBot/msagent.git
```

4. 安装msagent

```
cd msagent & pip install -e .
```

5. 执行测试例子

直接运行msagent指令，并通过config参数指定配置文件地址

```
msagent --config test.yaml
```


## 自定义算法


利用msagent的服务接口，编写服务脚本和环境脚本

msagent 提供了两套服务接口：服务请求接口`service`和事件相应接口`event_handler`

### 服务定义

```python
from msagent import Worker
# Worker是一个服务实例的抽象
class Service(Worker):

  def __init__(self, **kwargs):
    super().__init__() 
	
  # service装饰器定义了服务请求接口，可将类方法转化为远程服务，客户端通过remote装饰器获取该方法后，可以进行远程调用
  #   response : 默认为True, 当为False时将不回复，clients也不会阻塞（异步）
  #   batch : 默认为1, 当大于1时会同步等到batch个客户端的请求，以list的形式收集输入
  @service(response=False, batch=1)
  def func(self, *args, **kargs):
    pass

  # event_handler定义了事件响应接口，可将类方法转化为某事件响应句柄
  @event_handler(event_name,)
  def handler(self, *args, **kargs):
    pass
```

需要注意：每个Service服务都应包含在一个脚本文件中，且一个脚本只包含一个服务（`Worker`）类

### 服务调用

```python
from msagent import EventDispatcher
# 使用remote装饰器，可以标记func为远程函数，进而可以通过调用func实现对远程服务的调用
@remote
def func():
  pass

# 触发事件的接口
ed = EventDispatcher()
ed.fire(event_name, payload) 
```


2. 编写配置脚本，指定服务和环境的参数，具体包括：

  - 指定每个服务的路径、启动数量和相关参数
  - 指定环境的路径和启动数量
  - 指定日志路径

3. 全部定义好后, 便可通过`msagent --config [配置文件名称]`的方式启动训练启动