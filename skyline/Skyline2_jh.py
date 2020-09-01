#!/usr/bin/env python
# coding: utf-8

# In[133]:


import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
from collections import defaultdict

Skylines =[]
data_cnt = 25000
randnum = 200
# f_cnt=0
# a_cnt = 1
# l_cnt =0
def eu(x,y):
    n= x**2 + y**2
    return math.sqrt(n)


# In[149]:


#저널논문 - fig4 - 음의 상관관계
count=0
data = []
skyline = []
unique = []
f = open('negative-cor-10000000.txt', 'r')
lines = f.readlines()
for line in lines:
    item = line.split(" ")
    x = float(item[0])
    y = float(item[1])
    data.append(list((math.trunc(x*10)+41,math.trunc(y*10)+43)))

one = defaultdict(lambda : randnum)
two = defaultdict(lambda : randnum)
# for x,y in temp:
#     data.append(list((x,y)))
sCandidate = {} 

print("data길이: ",len(data))

# 중복체크
ex_list = list(set(map(tuple,data)))
print(len(ex_list))


timelap = time.time()

for x, y in ex_list:  
    if y < one[x]:
        one[x] = y 
        if two[one[x]] > x:
            two[one[x]] = x
            if two[one[x]] != randnum and two[one[x]] in sCandidate:
                del sCandidate[two[one[x]]]
            sCandidate[two[one[x]]] = one[x]
        elif two[one[x]] < x:   #여기 문제는 아닌것 같아 전혀 지장 없음
            if x in sCandidate:
                del sCandidate[x]   

for x,y in sCandidate.items():
    skyline.append([x,y])
    

def euclidean_distance(x,y):
    return math.sqrt(x**2+y**2)

############################# NN 인덱스 반환
def boundedNNSearch(data, distance_function):
    data_distance = []
    for x, y in data:
        data_distance.append(distance_function(x, y)) # 원점과 현재 데이터의 거리 계산 후 data_distance에 append한다
    
    nn_idx = data_distance.index(min(data_distance)) # Nearest Neighbor의 인덱스를 얻는다
    return nn_idx



############################# #Region을 0, 1, 2으로 나눠주는 함수
def data_cut(data, todo, nn, region_idx):
    region = []
    for i in range(len(data)):
        x, y = data[i]

        region_dict ={0: x > nn[0] and y > nn[1], 1: y > nn[1], 2: x > nn[0]} #Region 0인지 1인지 2인지 딕셔너리
            
        if region_dict[region_idx]:
            continue
        if x != nn[0] and y != nn[1]:
            region.append(data[i])
        
    if region_idx == 0:
        return region #Region의 데이터 반환

    else:
        if len(region) != 0: 
            todo.append(region) #해야할 일에 Region의 데이터 추가


############################# 메인 함수
def getskylines(data, distance_function=euclidean_distance):
    todo = []
    skylines = []
    todo.append(data) # Todo에 데이터 추가
    del data


    #### Todo 만큼 반복. Todo를 추가하면 추가된Todo 실행
    for i, region in enumerate(todo): #n
        nn = region[boundedNNSearch(region, distance_function)] ## NN 구하기
        
        data = data_cut(region, todo, nn, 0) # Region 3을 제외한 데이터 반환
        data_cut(data, todo, nn, 1)  # Region 1 을 todo에 추가
        data_cut(data, todo, nn, 2) # Region 2 todo에 추가

        skylines.append(nn) #스카이라인 추가
        print(nn)

        todo[i] = None # 필요없는 변수 없애기
    return skylines

answer = getskylines(skyline)

print("시간",time.time()-timelap)
print("skyline의 길이: ",len(answer))

put = np.array(ex_list)
skyline = np.array(answer)
plt.scatter(put[:,0],put[:,1], color="blue")
plt.scatter(skyline[:,0], skyline[:,1], color="red", label="sCandidate")
# plt.xlim(0, 18) 
# plt.ylim(0, 18)
plt.show()


# In[143]:


#저널논문 - fig4 - 양의 상관관계
count=0
data = []
skyline = []
unique = []
f = open('positive-cor-10000000.txt', 'r')
lines = f.readlines()
for line in lines:
    item = line.split(" ")
    x = float(item[0])
    y = float(item[1])
    data.append(list((math.trunc(x*10)+41,math.trunc(y*10)+43)))

one = defaultdict(lambda : randnum)
two = defaultdict(lambda : randnum)
# for x,y in temp:
#     data.append(list((x,y)))
sCandidate = {} 

print("data길이: ",len(data))

# 중복체크
ex_list = list(set(map(tuple,data)))
print(len(ex_list))


timelap = time.time()

for x, y in ex_list:  
    if y < one[x]:
        one[x] = y 
        if two[one[x]] > x:
            two[one[x]] = x
            if two[one[x]] != randnum and two[one[x]] in sCandidate:
                del sCandidate[two[one[x]]]
            sCandidate[two[one[x]]] = one[x]
        elif two[one[x]] < x:   #여기 문제는 아닌것 같아 전혀 지장 없음
            if x in sCandidate:
                del sCandidate[x]   

for x,y in sCandidate.items():
    skyline.append([x,y])
    

def euclidean_distance(x,y):
    return math.sqrt(x**2+y**2)

############################# NN 인덱스 반환
def boundedNNSearch(data, distance_function):
    data_distance = []
    for x, y in data:
        data_distance.append(distance_function(x, y)) # 원점과 현재 데이터의 거리 계산 후 data_distance에 append한다
    
    nn_idx = data_distance.index(min(data_distance)) # Nearest Neighbor의 인덱스를 얻는다
    return nn_idx



############################# #Region을 0, 1, 2으로 나눠주는 함수
def data_cut(data, todo, nn, region_idx):
    region = []
    for i in range(len(data)):
        x, y = data[i]

        region_dict ={0: x > nn[0] and y > nn[1], 1: y > nn[1], 2: x > nn[0]} #Region 0인지 1인지 2인지 딕셔너리
            
        if region_dict[region_idx]:
            continue
        if x != nn[0] and y != nn[1]:
            region.append(data[i])
        
    if region_idx == 0:
        return region #Region의 데이터 반환

    else:
        if len(region) != 0: 
            todo.append(region) #해야할 일에 Region의 데이터 추가


############################# 메인 함수
def getskylines(data, distance_function=euclidean_distance):
    todo = []
    skylines = []
    todo.append(data) # Todo에 데이터 추가
    del data


    #### Todo 만큼 반복. Todo를 추가하면 추가된Todo 실행
    for i, region in enumerate(todo): #n
        nn = region[boundedNNSearch(region, distance_function)] ## NN 구하기
        
        data = data_cut(region, todo, nn, 0) # Region 3을 제외한 데이터 반환
        data_cut(data, todo, nn, 1)  # Region 1 을 todo에 추가
        data_cut(data, todo, nn, 2) # Region 2 todo에 추가

        skylines.append(nn) #스카이라인 추가
        print(nn)

        todo[i] = None # 필요없는 변수 없애기
    return skylines

answer = getskylines(skyline)

print("시간",time.time()-timelap)
print("skyline의 길이: ",len(answer))

put = np.array(ex_list)
skyline = np.array(answer)
plt.scatter(put[:,0],put[:,1], color="blue")
plt.scatter(skyline[:,0], skyline[:,1], color="red", label="sCandidate")
# plt.xlim(0, 18) 
# plt.ylim(0, 18)
plt.show()


# In[131]:


#저널논문 - fig4 - 음의 상관관계
count=0
data = []
skyline = []
unique = []
f = open('imsi2.txt', 'r')
lines = f.readlines()
for line in lines:
    item = line.split(" ")
    x = float(item[0])
    y = float(item[1])
    data.append(list((math.trunc(x*10)+41,math.trunc(y*10)+43)))

one = defaultdict(lambda : randnum)
two = defaultdict(lambda : randnum)
# for x,y in temp:
#     data.append(list((x,y)))
sCandidate = {} 

print("data길이: ",len(data))

# 중복체크
ex_list = list(set(map(tuple,data)))
print(len(ex_list))


timelap = time.time()

def euclidean_distance(x,y):
    return math.sqrt(x**2+y**2)

############################# NN 인덱스 반환
def boundedNNSearch(data, distance_function):
    data_distance = []
    for x, y in data:
        data_distance.append(distance_function(x, y)) # 원점과 현재 데이터의 거리 계산 후 data_distance에 append한다
    
    nn_idx = data_distance.index(min(data_distance)) # Nearest Neighbor의 인덱스를 얻는다
    return nn_idx



############################# #Region을 0, 1, 2으로 나눠주는 함수
def data_cut(data, todo, nn, region_idx):
    region = []
    for i in range(len(data)):
        x, y = data[i]

        region_dict ={0: x > nn[0] and y > nn[1], 1: y > nn[1], 2: x > nn[0]} #Region 0인지 1인지 2인지 딕셔너리
            
        if region_dict[region_idx]:
            continue
        if x != nn[0] and y != nn[1]:
            region.append(data[i])
        
    if region_idx == 0:
        return region #Region의 데이터 반환

    else:
        if len(region) != 0: 
            todo.append(region) #해야할 일에 Region의 데이터 추가


############################# 메인 함수
def getskylines(data, distance_function=euclidean_distance):
    todo = []
    skylines = []
    todo.append(data) # Todo에 데이터 추가
    del data


    #### Todo 만큼 반복. Todo를 추가하면 추가된Todo 실행
    for i, region in enumerate(todo): #n
        nn = region[boundedNNSearch(region, distance_function)] ## NN 구하기
        
        data = data_cut(region, todo, nn, 0) # Region 3을 제외한 데이터 반환
        data_cut(data, todo, nn, 1)  # Region 1 을 todo에 추가
        data_cut(data, todo, nn, 2) # Region 2 todo에 추가

        skylines.append(nn) #스카이라인 추가
        print(nn)

        todo[i] = None # 필요없는 변수 없애기
    return skylines

answer = getskylines(ex_list)

print("시간",time.time()-timelap)
print("skyline의 길이: ",len(answer))

put = np.array(ex_list)
skyline = np.array(answer)
plt.scatter(put[:,0],put[:,1], color="blue")
plt.scatter(skyline[:,0], skyline[:,1], color="red", label="sCandidate")
# plt.xlim(0, 18) 
# plt.ylim(0, 18)
plt.show()


# In[128]:


#저널논문 - fig3 - 표준 분포
#random
np.random.seed(4)
distance = np.random.randint(3,randnum,size=data_cnt)
prices = np.random.randint(3,randnum,size=data_cnt)
temp = zip(distance,prices)
count=0  
data = []
skyline = []
unique = []
one = defaultdict(lambda : randnum)
two = defaultdict(lambda : randnum)

for x,y in temp:
    data.append(list((x,y)))
sCandidate = {}

print("data길이: ",len(data))
#중복체크
ex_list = list(set(map(tuple,data)))
print("중복체크 길이: ",len(ex_list))


timelap = time.time()
# dominant되지 않는 tuple들을 나타내기
for x, y  in ex_list:  
    if y < one[x]:
        one[x] = y 
        if two[one[x]] > x: 
            if two[one[x]] != randnum and two[one[x]] in sCandidate:
                del sCandidate[two[one[x]]]    
            two[one[x]] = x
            sCandidate[two[one[x]]] = one[x]
        elif two[one[x]] < x: 
            if x in sCandidate:
                del sCandidate[x]   
                
for x,y in sCandidate.items():
    skyline.append([x,y])
# print("개수: ", len(skyline))

def euclidean_distance(x,y):
    return math.sqrt(x**2+y**2)

############################# NN 인덱스 반환
def boundedNNSearch(data, distance_function):
    data_distance = []
    for x, y in data:
        data_distance.append(distance_function(x, y)) # 원점과 현재 데이터의 거리 계산 후 data_distance에 append한다
    
    nn_idx = data_distance.index(min(data_distance)) # Nearest Neighbor의 인덱스를 얻는다
    return nn_idx



############################# #Region을 0, 1, 2으로 나눠주는 함수
def data_cut(data, todo, nn, region_idx):
    region = []
    for i in range(len(data)):
        x, y = data[i]

        region_dict ={0: x > nn[0] and y > nn[1], 1: y > nn[1], 2: x > nn[0]} #Region 0인지 1인지 2인지 딕셔너리
            
        if region_dict[region_idx]:
            continue
        if x != nn[0] and y != nn[1]:
            region.append(data[i])
        
    if region_idx == 0:
        return region #Region의 데이터 반환

    else:
        if len(region) != 0: 
            todo.append(region) #해야할 일에 Region의 데이터 추가


############################# 메인 함수
def getskylines(data, distance_function=euclidean_distance):
    todo = []
    skylines = []
    todo.append(data) # Todo에 데이터 추가
    del data


    #### Todo 만큼 반복. Todo를 추가하면 추가된Todo 실행
    for i, region in enumerate(todo): #n
        nn = region[boundedNNSearch(region, distance_function)] ## NN 구하기
        
        data = data_cut(region, todo, nn, 0) # Region 3을 제외한 데이터 반환
        data_cut(data, todo, nn, 1)  # Region 1 을 todo에 추가
        data_cut(data, todo, nn, 2) # Region 2 todo에 추가

        skylines.append(nn) #스카이라인 추가
        print(nn)

        todo[i] = None # 필요없는 변수 없애기
    return skylines

answer = getskylines(skyline)
print("시간",time.time()-timelap)
print("skyline의 길이: ",len(answer))
print("unique의 길이: ",len(unique))
# print(ex_list, end = "\n\n")
print(answer)
skyline = np.array(answer)
plt.scatter(distance,prices, color="blue")
plt.scatter(skyline[:,0], skyline[:,1], color="red", label="sCandidate")
# plt.xlim(0, 10) 
# plt.ylim(0, 10)
plt.show()


# In[ ]:





# In[ ]:




