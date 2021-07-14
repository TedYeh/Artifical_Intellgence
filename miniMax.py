import random

def init_nodes():#隨機初始節點
    branch = eval(input('Branch：'))
    layers = eval(input('Layers：'))
    tree_nodes = [random.randint(-100, 100) for _ in range(branch ** (layers))]
    return tree_nodes, branch, layers

def cluster(tree_nodes, branch):# 分群
    clusters = []
    n = -1
    for i in range(len(tree_nodes)):
        if i % branch == 0:
            n += 1
            clusters.append([])
        clusters[n].append(tree_nodes[i])
    return clusters

def miniMax(clusters, layers, branch):
    flag = True if layers % 2 == 0 else False
    ans = list(clusters)
    print(ans)
    tmp = []
    tmp2, n = [], -1
    while len(tmp2) != 1 and len(tmp) != branch:        
        tmp = []
        tmp2, n = [], -1
        for i in range(len(ans)):
            if flag:tmp.append(min(ans[i]))
            else:tmp.append(max(ans[i]))
        tmp2 = cluster(tmp, branch)
        ans = list(tmp2)
        print(ans)
        flag = not flag
        
    return max(tmp)
p = 0
travel = []
MIN, MAX = -999, 999
def miniMax_alpha_beta(layers, branch, depth, index, nodes, isPlayer, beta, alpha):
        
    if depth == layers:return nodes[index]
    
    #AI playing:choose the big one
    if isPlayer:
        best = MIN
        for i in range(0, branch):            
            val = miniMax_alpha_beta(layers, branch ,depth+1, index*branch+i, nodes, False, beta, alpha) #get player state
            #print(nodes)
            best = max(best, val)
            alpha = max(alpha, best)  
            if beta <= alpha:break
            travel.append(nodes[index*branch+i])
        travel.append(nodes[index*branch+i])
        return best          
    else:
    #player playing:choose the small one
        best = MAX
        for i in range(0, branch):
            val = miniMax_alpha_beta(layers, branch ,depth+1, index*branch+i, nodes, True, beta, alpha) #get AI state
            #print(nodes)
            best = min(best, val)
            beta = min(beta, best)  
            if beta <= alpha:break
        return best          
    
#def alpha_beta_for_loop():


if __name__ == '__main__':
    '''
    nodes, branch, layers = init_nodes()
    #nodes = [11,-2,3,6,-8,9,15,4]
    clusters = cluster(nodes, branch)
    print(nodes)
    ans = miniMax(clusters, layers, branch)
    print(ans)
    #print(len(nodes), clusters, ans)

    '''
    fileName = input('input the file name(ex. test.txt)：')
    with open(fileName, 'r', encoding='utf-8') as test:
            branches, layers, nodes = test.read().split('\n')
            branches, layers, nodes = int(branches), int(layers), [int(i) for i in nodes.split(',')]
            print(miniMax_alpha_beta(layers, branches, 0, 0, nodes, True, MAX, MIN))
            t = list(set(travel))
            #t.sort()
            for i in range(1, len(nodes)+1):
                if i not in travel:print(i, end=' ')
            #print(p)
    '''
    try:
        
    except:
        print('file name format ERROR！')
    '''
    
    
