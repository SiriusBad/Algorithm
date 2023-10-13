import time
import math

def Find_Max_Cross_Subarray(A,low,mid,high):
    array = {}
    left_sum = -math.inf
    sum = 0
    for i in range(mid,low - 1,-1):
        sum = sum + A[i]
        if sum > left_sum:
            left_sum = sum
            max_left = i

    right_sum = -math.inf
    sum = 0
    for j in range(mid + 1, high + 1):
        sum = sum + A[j]
        if sum > right_sum:
            right_sum = sum
            max_right = j

    array['left'] = max_left
    array['right'] = max_right
    array['sum'] = left_sum + right_sum
    return array

def Find_Maximum_Subarray(A,low,high):
    array = {}
    if high == low:
        array['left'] = low
        array['right'] = high
        array['sum'] = A[low]
        return array
    else:
        mid = int((low+high)/2)
        array_left = Find_Maximum_Subarray(A,low,mid)
        array_right = Find_Maximum_Subarray(A,mid+1,high)
        array_mid = Find_Max_Cross_Subarray(A,low,mid,high)
        if array_left['sum']>array_right['sum'] and array_left['sum']>array_mid['sum']:
            return array_left
        elif array_right['sum']>array_left['sum'] and array_right['sum']>array_mid['sum']:
            return array_right
        else:
            return array_mid

A = [13,-3,-25,20,-3,-16,-23,18,20,-7,12,-5,-22,15,-4,7]
print(Find_Maximum_Subarray(A,0,len(A)-1))



