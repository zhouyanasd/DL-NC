class Base(object):

    __global_time = 0
    __global_connection = 0

    def get_global_time(self):
        return self.get_global_time

    def set_global_time(self,time):
        self.__global_time = time

    def add_global_time(self,dt = 1):
        self.__global_time = self.__global_time + dt