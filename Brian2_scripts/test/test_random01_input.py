from brian2 import *

def patterns_classification(duration, patterns, neu =1, interval_l=10, interval_s = ms):
    def tran_patterns(A, patterns):
        trans = []
        for a in A:
            for i in range(int(interval_l/2)):
                trans.append(0)
            a_ =patterns[a]
            for i in a_:
                trans.append(int(i))
            for i in range(int(interval_l/2)):
                trans.append(0)
        return np.asarray(trans)
    interval = interval_l + patterns.shape[1]
    if (duration/interval_s) % interval != 0:
        raise ("duration and interval+len(patterns) must be exacted division")
    n = int((duration/interval_s)/interval)
    label = np.random.randint(0,int(patterns.shape[0]),n)
    seq = tran_patterns(label,patterns)
    times = where(seq ==1)[0]*interval_s
    indices = zeros(int(len(times)))
    return times, seq
    # P = SpikeGeneratorGroup(neu, indices, times)
    # return P , label

patterns = np.array([[1,1,1,1,1,0,0,0,0,0],
                     [1,0,1,1,0,0,1,0,1,0],
                     [1,0,1,1,0,1,0,0,0,1]])

t,s = patterns_classification(100*ms, patterns, interval_l=10)

print(s)