from brian2 import *

def binary_classification(duration, start=1, end =7, neu =1, interval_l=10, interval_s = ms):
    def tran_bin(A):
        trans = []
        for a in A:
            for i in range(13):
                trans.append(0)
            a_ = bin(a)[2:]
            while len(a_) <3:
                a_ = '0'+a_
            for i in a_:
                trans.append(int(i))
            for i in range(4):
                trans.append(0)
        return np.asarray(trans)
    def tran_bin_hard(A, Patterns):
        trans = []
        for a in A:
            for i in range(13):
                trans.append(0)
            a_ =Patterns[a]
            for i in a_:
                trans.append(int(i))
            for i in range(4):
                trans.append(0)
        return np.asarray(trans)
    n = int((duration/interval_s)/interval_l)
    label = np.random.randint(start,end,n)
    seq = tran_bin(label)
    times = where(seq ==1)[0]*interval_s
    indices = zeros(int(len(times)))
    P = SpikeGeneratorGroup(neu, indices, times)
    return P , label
