from time import time
from multiprocessing import Pool
def fib(x):
    if x<2:
        return 1
    return fib(x-1)+fib(x-2)
myTimer=time()
# print(fib(36), time()-myTimer)
# print(fib(36), time()-myTimer)
myPool=Pool()
result1=myPool.apply_async(fib, [36])
result2=myPool.apply_async(fib, [36])
print(result1.get(),result2.get(),time()-myTimer)