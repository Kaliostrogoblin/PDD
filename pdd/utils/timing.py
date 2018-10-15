from time import time

def timeit(f):
    def wrapper(*args, **kwargs):
        start = time()
        res = f(*args, **kwargs)
        print("--[%s] took %.2f seconds to run.\n" % 
            (f.__name__, time() - start))
        return res
    return wrapper