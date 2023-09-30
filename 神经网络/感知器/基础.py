# 与门
def AND(x1, x2):
    w1, w2 = 0.3, 0.3  # 权值
    temp = 0.5  # 阈值
    n = w1 * x1 + w2 * x2
    if n > temp:
        return 1
    else:
        return 0


# 非门
def NAND(x1, x2):
    w1, w2 = 0.3, 0.3
    temp = 0.2
    n = w1 * x1 + w2 * x2
    if n < temp:
        return 1
    else:
        return 0


# 或门
def OR(x1, x2):
    w1, w2 = 0.3, 0.3
    temp = 0.2
    n = w1 * x1 + w2 * x2
    if n > temp:
        return 1
    else:
        return 0
