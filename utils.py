from typing import List

def size_of_list(list: List):# 表中所有元素的乘积
    result = 1
    for i in list:
        result *= i
    return result

def size(list):# 如果输入是列表，返回列表中所有元素的乘积；如果输入是对象，返回对象的 size 属性
    if isinstance(list, List):     
        return size_of_list(list)
    else:
        return list.size

def closest_factors(n):# 寻找给定整数 n 的最接近的两个因数  比如传入12返回3,4
    x = int(n**0.5)
    while x >= 1:
        if n % x == 0:
            return x, n // x
        x -= 1
    return 0,0