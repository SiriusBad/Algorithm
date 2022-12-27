def Insertion_Sort(A):
    for i in range(1,len(A)):
        key = A[i]
        j = i - 1
        while j>=0 and A[j]>key:
            A[j+1] = A[j]
            j -= 1
        A[j+1] = key
    return A

print(Insertion_Sort([1,8,3,6,32,7,3,5,7,3,746,34,2477,34,9]))
