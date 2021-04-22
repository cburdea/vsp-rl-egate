

import multiprocessing
def sum_up_to(number):
    return sum(range(1, number + 1))


a_pool = multiprocessing.Pool()


result = a_pool.map(sum_up_to, range(10))

print(result)
