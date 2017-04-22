from brian2 import *
start_scope()

stimulus = TimedArray(np.hstack([[c, c, c, 0, 0]
                                 for c in np.random.rand(1000)]),
                                dt=10*ms)

equ1 = '''
dv/dt = (-v + stimulus(t)*mV)/(10*ms) : volt (unless refractory)
'''

equ2 = '''
dv/dt = (-v + I)/(10*ms) : volt (unless refractory)
I : volt
'''

equ3 ='''
dv/dt = (-v + I)/(10*ms) : 1
rates : Hz  # each neuron's input has a different rate
size : 1  # and a different amplitude
I = size*(sin(2*pi*rates*t)+1) : 1
'''

G1 = NeuronGroup(1, equ1,threshold='v > 0.9*mV', reset='v = 0*mV',
                    refractory=1*ms, method='linear')

G2 = NeuronGroup(1, equ2,threshold='v > 0.9*mV', reset='v = 0*mV',
                    refractory=1*ms, method='linear')

G2.run_regularly('''change = int(rand() < 0.5)
                   I = change*(rand()*2)*mV + (1-change)*I''',
                dt=50*ms)

G3 = NeuronGroup(2, equ3,method='euler',threshold='v > 0.9', reset='v = 0',refractory=1*ms)
G3.rates = '10*Hz + i*Hz'
G3.size = '(100-i)/100. + 0.1'

m1=StateMonitor(G3,'I',record=0)
m2=StateMonitor(G3,'v',record=0)

run(1000*ms)
plt.figure(figsize = (20,4))
subplot(121)
plot(m1.t/ms,m1.I[0],'-b', label='Neuron 0')
subplot(122)
plot(m2.t/ms,m2.v[0],'-b', label='Neuron 0')
show()