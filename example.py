T= int(input())
li1 =[]
for i in range(T):
    li = input().split(" ")
    train = li[0]
    time = li[-1]
    place = li[-3]
    idx = i
    if time[0] == 0:
      hour = time[0]
    else:
      hour = time[0:2]
    in_li = [train,place , time , idx]
    li1.append(in_li)
    #li2 = sorted(li1 , key = lambda li: (li[0],li[3]))


def bubbleSort(arr):
    for i in range(len(arr)-1):
        flag = 'no_swap'
        for j in range(len(arr)-1-i):
            if arr[j][0] > arr[j+1][0]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
            if arr[j][0] == arr[j+1][0] and arr[j][2] < arr[j+1][2]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
            flag = 'swapped'

        if flag == 'no_swap':
            break
    return arr

Train_p = bubbleSort(li1)

for i in Train_p:
    print(f"{i[0]} will departure for {i[1]} at {i[2]}")