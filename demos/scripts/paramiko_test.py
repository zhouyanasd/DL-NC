from time import *
import paramiko


# 定义一个类，表示一台远端linux主机
class Linux(object):
    # 通过IP, 用户名，密码，超时时间初始化一个远程Linux主机
    def __init__(self, ip, username, password, timeout=30):
        self.ip = ip
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
                self.t = paramiko.Transport(sock=(self.ip, 11179))
                self.t.connect(username=self.username, password=self.password)
                self.chan = self.t.open_session()
                self.chan.settimeout(self.timeout)
                self.chan.get_pty()
                self.chan.invoke_shell()
                # 如果没有抛出异常说明连接成功，直接返回
                print('连接%s成功' % self.ip)
                # 接收到的网络数据解码为str
                print(self.chan.recv(65535).decode('utf-8'))
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
            sleep(0.5)
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


# 连接正常的情况
if __name__ == '__main__':
    host = Linux('219.216.81.148', 'zy', '88507840')  # 传入Ip，用户名，密码
    host.connect()


    # result = host.send('ls') # 发送一个查看ip的命令
    def input_cmd(str):
        return input(str)


    tishi_msg = "输入命令："
    while True:
        msg = input(tishi_msg)
        if msg == "exit":
            host.close()
            break
        else:
            res = host.send(msg)
            data = res.replace(res.split("\n")[-1], "")
            tishi_msg = res.split("\n")[-1]
            print(res.split("\n")[-1] + data.strip("\n"))
