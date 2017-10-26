class a():
    def __init__(self):
        self.t = 1
    def aa(self):
        print('a')

class b(a):
    def __init__(self):
        a.__init__(self)
        self.tt=2
    def aa(self):
        print('b')
        super(b, self).aa()

B=b()
B.aa()
print(B.tt)