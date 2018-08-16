from time import time

def timeit(f):
    def wrapper(*args, **kwargs):
        start = time()
        f(*args, **kwargs)
        print("Took %.2f seconds to run.\n" % (time() - start))
    return wrapper