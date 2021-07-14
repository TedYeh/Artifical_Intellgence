import matplotlib.pyplot as plt
import numpy as np
import random
import time

def init(q_num): #起始狀態
    '''
    return list -->索引為行，值為列
    ex.
  col  0 1 2 3 4 5 6 7
  row [0 1 2 3 4 5 6 7] (list)
    '''
    status = [i for i in range(q_num)]
    random.shuffle(status)
    return status

def conflict(status): #是否衝突
    n = 0
    for i in range(len(status)):
        for j in range(i + 1, len(status)):
            if status[i] == status[j]:n += 1 #same row
            if abs(status[i] - status[j]) == abs(i - j):n += 1 #對角線
    return n

def neighbor(status): #算鄰居節點
    neig = {}
    #current = conflict(status)
    for i in range(len(status)):
        for j in range(len(status)):
            if status[i] == j:continue #不將自己加入考慮
            tmp = list(status)
            tmp[i] = j
            neig[(i, j)] = conflict(tmp)
    return neig #回傳每個鄰居的衝突個數

def hill_climbing(status, neig): #爬山算法
    optima = []
    current = conflict(status)
    #紀錄部份最佳解
    for key, value in neig.items():
        if value <= current: current = value
    for key, value in neig.items():
        if value == current: optima.append(key)
    #若最佳解超過1個，隨便選一個
    if len(optima) > 0:
        next_node = random.choice(optima)
        status[next_node[0]] = next_node[1]

    return status

def K_Queen(status): #直到沒有衝突為止
    current = conflict(status)
    n = 0
    max_time = 1000
    while current > 0 and n <= max_time:
        neig = neighbor(status)
        status = hill_climbing(status, neig)
        current = conflict(status)
        n += 1
    return status, n

def write_file(status):
    with open('1106108121_葉丞鴻.txt', 'w', encoding='utf-8') as f:
        f.write(f'{len(status)}\n')
        for i in range(len(status)):
            f.write(f'{i},{status[i]}\n')

def show_board(status):
    x = np.array([i+0.5 for i in range(len(status))])
    y = np.array([i+0.5 for i in status])
    size = np.arange(0, len(status)+1)
    plt.plot(x, y, 'ro')
    plt.xticks(size)
    plt.yticks(size)
    plt.grid()
    plt.show()

def show_all_answer(k):
    status = []
    f = open('all_answers.txt', 'w', encoding='utf-8')
    while len(status) < 92:
        init_status = init(k)
        #print(init_status)
        optima, n = K_Queen(init_status)
        while n >= 1000:
            print('restart')            
            init_status = init(k)
            optima, n = K_Queen(init_status)

        if not optima in status and conflict(optima)==0:
            status.append(optima)
    f.write(f'{len(status)}\n')
    for i in range(len(status)):
        s = ''
        for point in status[i]:s += str(point) + ' '
        f.write(f'{s}\n')
            
def show_one_answer(k):
    init_status = init(k)            
    print("init：", init_status)
    optima, n = K_Queen(init_status)

    while n >= 1000:
        print('restart')
        init_status = init(k)
        optima, n = K_Queen(init_status)

    print("final：", optima)
    write_file(optima)
    show_board(optima)

if __name__ == '__main__':
    mode = input('which mode you want to test？\n1) show one answer \n2) show all answers\n：')
    if eval(mode) == 1:
        k = eval(input('How many Queens do you want ?'))
        show_one_answer(k)
    else:   
        k = eval(input('How many Queens do you want ?'))    
        show_all_answer(k)   
    
    