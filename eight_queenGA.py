import matplotlib.pyplot as plt
import numpy as np
import random
import time

'''
initial : Generate n herd
crossover : if rand < prob_c : change gen
mutation : if rand < prob_m : mutate gen

'''

def encode(pop):
    new_pop, n = list(), 0
    nums = [i for i in range(len(pop))]
    while True:
        if n == len(pop):break
        want = pop[n]
        pos = nums.index(want)
        new_pop.append(pos)
        n += 1
        nums.pop(pos)
    return new_pop

def decode(pop):
    old_pop, n = list(), 0
    nums = [i for i in range(len(pop))]
    for n in range(len(pop)):
        pos = pop[n]
        want = nums[pos]
        old_pop.append(want)
        nums.pop(pos)
    return old_pop

def init(q_num, herd): #起始狀態
    '''
    return list -->索引為行，值為列
    ex.
  col  0 1 2 3 4 5 6 7
  row [0 1 2 3 4 5 6 7] (list)
    '''
    status = [[i for i in range(q_num)]for _ in range(herd)]
    for i in range(herd):random.shuffle(status[i])
    random.shuffle(status)
    #status = np.array(status)
    return status

def conflict(status): #fitness function，檢查是否衝突
    n = 0
    for i in range(len(status)):
        for j in range(i + 1, len(status)):
            if status[i] == status[j]:n += 1 #same row
            if abs(status[i] - status[j]) == abs(i - j):n += 1 #對角線
    return n

def selection(status, scores, k=3): # select the best
    select_ix = np.random.randint(len(status))
    for ix in np.random.randint(0, len(status), k-1):
        if scores[ix] < scores[select_ix]:
            select_ix = ix
    return status[select_ix]

def mutation(pop, p_mut):
    if random.random() < p_mut:
        n1, n2 = np.random.randint(len(pop)), np.random.randint(len(pop))
        while n1 == n2:n2 = np.random.randint(len(pop))
        pop[n1], pop[n2] =  pop[n2], pop[n1]
    return pop

def crossover(s1, s2, p_cro):
    tmp1, tmp2 = list(s1), list(s2) 
    if random.random() < p_cro:
        pos = np.random.randint(len(s1))
        tmp1 = s1[:pos] + s2[pos:]
        tmp2 = s2[:pos] + s1[pos:] 
    return [tmp1, tmp2]

def KqueenGA(q_num, herd, n_iter, p_cro, p_mut): #直到沒有衝突為止
    #initial population 
    restart_time = 0
    pops = init(q_num, herd) #generate n pairs of parents
    gen = 0
    #default best status
    best, best_score = [], conflict(pops[0])

    while True:
        scores = [conflict(p) for p in pops] #record each population's score(conflict)

        # display new best solution
        for i in range(herd):
            if scores[i] < best_score:
                best, best_score = pops[i], scores[i]
                print(">%d, new best f(%s) = %2d" % (gen,  best, best_score))
        if best_score == 0:break
        if gen >= 1000:
            #restart
            restart_time += 1
            print('restart')
            pops, gen, best = init(q_num, herd), 0, [] 
            best_score = conflict(pops[0])

        #select parents
        #selected = [selection(pops, scores) for _ in range(herd)] if herd % 2==0 else [selection(pops, scores) for _ in range(herd+1)]
        selected = [selection(pops, scores) for _ in range(herd+1)]
        
        children = list()
        for i in range(0, herd, 2):
            p1, p2 = selected[i], selected[i+1]            

            for c in crossover(p1, p2, p_cro):                
                c = mutation(c, p_mut)
                children.append(c)

        pops = children
        gen += 1
    print(">%d, new best f(%s) = %2d | restart %2d times" % (gen,  best, best_score, restart_time))
    return [best, best_score]

def write_file(status):
    with open('1106108121_葉丞鴻.txt', 'w', encoding='utf-8') as f:
        f.write(f'{len(status)}\n')
        for i in range(len(status)):
            f.write(f'{i}, {status[i]}\n')

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
    '''
    mode = input('which mode you want to test？\n1) show one answer \n2) show all answers\n：')
    if eval(mode) == 1:
        k = eval(input('How many Queens do you want ?'))
        show_one_answer(k)
    else:   
        k = eval(input('How many Queens do you want ?'))    
        show_all_answer(k)   
    
    status = init(8, 12)
    for i in status:
        print(conflict(i))
    '''
    
    try: 
        k = eval(input('How many Queens do you want ?'))
        herd = eval(input('How many herd do you want ?'))
        gen = eval(input('gen ?'))
        best, best_score = KqueenGA(k, herd, gen, 0.9, 0.001)
        write_file(best)
        show_board(best)
    except:
        print('input format ERROR\n OR UNexcept ERROR')
    '''
    pop = [0, 1, 2, 3, 4, 5, 6, 7]
    random.shuffle(pop)
    print(pop)
    new_pop = encode(pop)
    print(new_pop)
    print(decode(new_pop))
    '''
    