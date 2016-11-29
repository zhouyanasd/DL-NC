import time

class Base(object):
    global_time = 0
    print("base: ",global_time)

class implement(Base):

    def print(self):
        for i in range(0,10,1):
            print("implement: ",i,self.global_time)
            Base.global_time = 1


imp = implement()
imp.print()