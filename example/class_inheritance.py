class a():
    def aa(self):
        print('a')

class b(a):
    def aa(self):
        print('b')
        super(b, self).aa()

B=b()
B.aa()