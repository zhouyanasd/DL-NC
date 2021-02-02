import ray, time
ray.init('auto')

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0
        self.time = time.time()

    def increment(self):
        self.value += 1
        print(time.time() - self.time)
        return self.value

    def get_counter(self):
        return self.value

counter_actor = Counter.remote()

assert ray.get(counter_actor.increment.remote()) == 1

for _ in range(20):
    print(ray.get(counter_actor.increment.remote()))

# Create ten Counter actors.
counters = [Counter.remote() for _ in range(10)]

# Increment each Counter once and get the results. These tasks all happen in
# parallel.
results = ray.get([c.increment.remote() for c in counters])
print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Increment the first Counter five times. These tasks are executed serially
# and share state.
results = ray.get([counters[0].increment.remote() for _ in range(5)])
print(results)  # prints [2, 3, 4, 5, 6]

@ray.remote
class Foo(object):

    # Any method of the actor can return multiple object refs.
    @ray.method(num_returns=2)
    def bar(self):
        return 1, 2

f = Foo.remote()

obj_ref1, obj_ref2 = f.bar.remote()
assert ray.get(obj_ref1) == 1
assert ray.get(obj_ref2) == 2

@ray.remote
def ff(counter):
    for _ in range(1000):
        time.sleep(0.1)
        counter.increment.remote()

[ray.get(ff.remote(counter_actor)) for _ in range(3)]

# Print the counter value.
for _ in range(10):
    time.sleep(1)
    print(ray.get(counter_actor.get_counter.remote()))