import paramiko, requests, time
import ray
import numpy  as np

class Linux(object):
    # 通过IP, 用户名，密码，超时时间初始化一个远程Linux主机
    def __init__(self, ip, port, username, password, timeout=30):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        # transport和chanel
        self.t = ''
        self.chan = ''
        # 链接失败的重试次数
        self.try_times = 3

    # 调用该方法连接远程主机
    def connect(self):
        while True:
            # 连接过程中可能会抛出异常，比如网络不通、链接超时
            try:
                self.t = paramiko.Transport(sock=(self.ip, self.port))
                self.t.connect(username=self.username, password=self.password)
                self.chan = self.t.open_session()
                self.chan.settimeout(self.timeout)
                self.chan.get_pty()
                self.chan.invoke_shell()
                # 如果没有抛出异常说明连接成功，直接返回
                print('连接%s成功' % self.ip)
                # 接收到的网络数据解码为str
                # print(self.chan.recv(65535).decode('utf-8'))
                return
            # 这里不对可能的异常如socket.error, socket.timeout细化，直接一网打尽
            except Exception as e1:
                if self.try_times != 0:
                    print('连接%s失败，进行重试' % self.ip)
                    self.try_times -= 1
                else:
                    print('重试3次失败，结束程序')
                    exit(1)

    # 断开连接
    def close(self):
        self.chan.close()
        self.t.close()

    # 发送要执行的命令
    def send(self, cmd):
        cmd += '\r'
        result = ''
        # 发送要执行的命令
        self.chan.send(cmd)
        # 回显很长的命令可能执行较久，通过循环分批次取回回显,执行成功返回true,失败返回false
        while True:
            time.sleep(5)
            ret = self.chan.recv(65535)
            ret = ret.decode('utf-8')
            result += ret
            return result

    '''
    发送文件
    @:param upload_files上传文件路径 例如：/tmp/test.py
    @:param upload_path 上传到目标路径 例如：/tmp/test_new.py
    '''

    def upload_file(self, upload_files, upload_path):
        try:
            tran = paramiko.Transport(sock=(self.ip, self.port))
            tran.connect(username=self.username, password=self.password)
            sftp = paramiko.SFTPClient.from_transport(tran)
            result = sftp.put(upload_files, upload_path)
            return True if result else False
        except Exception as ex:
            print(ex)
            tran.close()
        finally:
            tran.close()

class Cluster():
    def __init__(self, cluster_config):
        self.head = cluster_config['head']
        self.nodes = cluster_config['nodes']

    def _execute(self, address, port, name, password, comments):
        linux = Linux(address, port, name, password)
        linux.connect()
        for msg in comments:
            linux.send(msg)
        linux.close()

    def start(self):
        head = Linux(self.head['manage_address'], self.head['manage_port'], self.head['name'], self.head['password'])
        head.connect()
        for msg in self.head['start_commends']:
            head.send(msg)
        self.check_dashboard('http:// ' +self.head['manage_address' ] +': ' +self.head['dashboard'])
        print('start head:  '+ self.head['manage_address' ] +': ' +str(self.head['manage_port']))
        head.close()

        for address, port, name, password in zip(self.nodes['manage_address_list'], self.nodes['manage_port_list'],
                                                 self.nodes['name_list'], self.nodes['password_list']):
            node = Linux(address, port, name, password)
            node.connect()
            for msg in self.nodes['start_commends']:
                node.send(msg)
            print('start node: ' + address + ':' + str(port))
            node.close()
        self.reconnect(self.check_alive())

    def stop(self):
        for address, port, name, password in zip(self.nodes['manage_address_list'], self.nodes['manage_port_list'],
                                                 self.nodes['name_list'], self.nodes['password_list']):
            node = Linux(address, port, name, password)
            node.connect()
            for msg in self.nodes['stop_commends']:
                node.send(msg)
            print('stop node: ' + address + ':' + str(port))
            node.close()
        head = Linux(self.head['manage_address'], self.head['manage_port'], self.head['name'], self.head['password'])
        head.connect()
        for msg in self.head['stop_commends']:
            head.send(msg)
        print('stop head: ' + self.head['manage_address'] + ':' + str(self.head['manage_port']))
        head.close()

    def check_dashboard(self, url):
        try_time = 5
        while try_time:
            try:
                requests.get(url)
                break
            except:
                time.sleep(1)
                try_time -= 1
                continue

    def check_alive(self):
        cluster_address = self.nodes['address_list'].copy()
        cluster_address.append(self.head['address'])
        mask = np.array([True] * len(cluster_address), dtype = np.bool)
        if not ray.is_initialized():
            ray.init('auto')
        for node in ray.nodes():
            if node['Alive'] is True and node['NodeManagerAddress'] in cluster_address:
                mask[cluster_address.index(node['NodeManagerAddress'])] = False
        dead_node = list(np.array(cluster_address)[mask])
        ray.shutdown()
        return dead_node

    def restart(self):
        print('cluster restart')
        self.stop()
        self.start()

    def reconnect(self, cluster_address):
        if len(cluster_address) <= 0:
            return
        for node_address in cluster_address:
            if node_address == self.head['address']:
                self.restart()
            if node_address in self.nodes['address_list']:
                print('reconnecting: ' + node_address)
                index = self.nodes['address_list'].index(node_address)
                self._execute(self.nodes['manage_address_list'][index], self.nodes['manage_port_list'][index],
                              self.nodes['name_list'][index], self.nodes['password_list'][index],
                              self.nodes['start_commends'])

    def init(self, **kwargs):
        try:
            kwargs['address']
        except KeyError:
            kwargs['address'] = 'auto'
        if ray.is_initialized():
            ray.shutdown()
        try:
            ray.init(**kwargs)
        except ConnectionError:
            self.start()