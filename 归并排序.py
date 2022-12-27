import math
Max = math.inf
def Merge(A,p,q,r):
    # p = 0
    # r = len(A)-1
    # q = int(len(A)/2)-1
    n1 = q-p+1
    n2 = r-q
    L = []
    R = []
    for i in range(n1):
        L.append(A[p+i])
    for i in range(n2):
        R.append(A[q+i+1])
    L.append(Max)
    R.append(Max)
    m = 0
    n = 0
    for i in range(p,r+1):
        if L[m]<=R[n]:
            A[i] = L[m]
            m += 1
        else:
            A[i] = R[n]
            n += 1
    return A

def Merge_Sort(A,p,r):
    if p<r:
        q = int((p+r)/2)
        Merge_Sort(A,p,q)
        Merge_Sort(A,q+1,r)
        Merge(A,p,q,r)
    return A
list = [1,4,2,6,3,64]
r = len(list)-1
print(Merge_Sort(list,0,r))