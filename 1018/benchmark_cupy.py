import numpy, cupy, time, os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

def test(xp):
    N = 2000
    a = xp.arange(N**2).reshape(N, N)
    start = time.time()
    a.dot(a)
    end = time.time()
    print('{xp} took {sec} sec.'.format(xp=xp.__name__, sec=end-start))

test(numpy)
test(cupy)
