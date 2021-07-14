import numpy as np
import time

class Reversi(object):
    POS = []
    def __init__(self, h, w):
        self.height = h
        self.width = w
        self.board = np.zeros((self.height, self.width))        

    def init_board(self):#初始化棋盤
        H, W = self.height//2, self.width//2
        self.board[H, W] = 2
        self.board[H - 1, W] = 1
        self.board[H, W - 1] = 1
        self.board[H -1, W - 1] = 2

    def is_on_board(self, x, y):#檢查棋子是否出界
        return (x < self.width and x >= 0) and (y < self.height and y >= 0)        

    def check_move_valid(self, xstart, ystart, player):
        valid_move = []     
        if not self.is_on_board(xstart, ystart) or self.board[xstart][ystart] != 0:
            #print('invalid',xstart, ystart)
            return valid_move

        direction = [(1, 1), (1, 0), (1, -1), (0, 1), (-1, -1), (0, -1), (-1, 0), (-1, 1)]
        otherplayer = 1 if player == 2 else 2
        self.board[xstart, ystart] = player
        for xdir, ydir in direction:
            x, y = xstart + xdir, ystart + ydir
            #print(x, y)
            while self.is_on_board(x, y) and self.board[x][y] == otherplayer:
                x += xdir
                y += ydir
                if self.is_on_board(x, y) and self.board[x][y] == player:
                    while True:
                        x -= xdir
                        y -= ydir
                        if x == xstart and y == ystart:break
                        valid_move.append((x, y))

        self.board[xstart][ystart] = 0
        return valid_move
    
    def get_valid_moves(self, player):
        return [[x, y] for x in range(self.width) for y in range(self.height) if self.check_move_valid(x, y, player)]

    def chess(self, player, xpos, ypos):
        valid_moves = self.check_move_valid(xpos, ypos, player)
        
        if valid_moves:
            self.board[xpos, ypos] = player
            for (x, y) in valid_moves:
                self.board[x, y] = player
        return valid_moves

    def get_num(self):
        num = {1:0, 2:0}
        for x in range(self.width):
            for y in range(self.height):
                tile = self.board[x][y]
                if tile in num.keys():
                    num[tile] += 1
        return num

    def get_weighted_score(self, player):
        SQUARE_WEIGHTS = {8:[
          120, -20,  20,   5,   5,  20, -20, 120,
          -20, -40,  -5,  -5,  -5,  -5, -40, -20,
           20,  -5,  15,   3,   3,  15,  -5,  20,
            5,  -5,   3,   3,   3,   3,  -5,   5,
            5,  -5,   3,   3,   3,   3,  -5,   5,
           20,  -5,  15,   3,   3,  15,  -5,  20,
           20, -40,  -5,  -5,  -5,  -5, -40, -20,
          120, -20,  20,   5,   5,  20, -20, 120,            
        ],
        10:[
            120, -20,  20,   5,   5,   5,   5,  20, -20, 120,
            -20, -40,  -5,  -5,  -5,  -5,  -5,  -5, -40, -20,
             20,  -5,  15,   0,   0,   0,   0,  15,  -5,  20,
              5,  -5,   0,   0,   0,   0,   0,   0,  -5,   5,  
              5,  -5,   0,   0,   0,   0,   0,   0,  -5,   5,
              5,  -5,   0,   0,   0,   0,   0,   0,  -5,   5,
              5,  -5,   0,   0,   0,   0,   0,   0,  -5,   5,
             20,  -5,  15,   0,   0,   0,   0,  15,  -5,  20,  
            -20, -40,  -5,  -5,  -5,  -5,  -5,  -5, -40, -20,
            120, -20,  20,   5,   5,   5,   5,  20, -20, 120,            
        ]}
        weights = SQUARE_WEIGHTS[self.height]
        
        total = 0
        other_player = 2 if player == 1 else 1
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x][y] == player:total += weights[x*8 + y]
                #elif self.board[x][y] == other_player:total += -1 * weights[x*8 + y]
        return total  
    
    def evaluate(self, player): #SBE
        #2*self.stability(player)
        #print(self.discDifference(player), self.score(player))
        #return 3*self.mobility(player) + 5*self.discDifference(player) 
        return 2*self.mobility(player) + 5*self.corners(player) + 4*self.discDifference(player) + 2*self.score(player) 
        #return 5*self.discDifference(player) + 3*self.get_num()[player]

    def score(self, player): #比分差(score matrix)
        my_score = self.get_weighted_score(player)
        other_score = self.get_weighted_score(2 if player == 1 else 1)
        return 100*(my_score - other_score) / max(my_score + other_score + 1, 1)

    def discDifference(self, player): #棋子差
        my_counts = self.get_num()[player]
        other_counts = self.get_num()[1 if player==2 else 2]        
        return 100 * (my_counts - other_counts) / (my_counts + other_counts + 1)        

    def corners(self, player): #角落數差
        corner = [[0, 0], [0, 7], [7, 0], [7, 7]]
        my_corners, other_corners = 0, 0
        for c in corner:
            if self.board[c[0], c[1]] == player:my_corners += 1
            elif self.board[c[0], c[1]] == (2 if player == 1 else 1): other_corners += 1
        return 100 * (my_corners - other_corners) / (my_corners + other_corners + 1)

    def mobility(self, player): #行動力
        my_moves = len(self.get_valid_moves(player))
        other_moves = len(self.get_valid_moves(2 if player == 1 else 1))
        #print(my_moves, other_moves, self.get_valid_moves(player))
        return 100 * (my_moves - other_moves) / (my_moves + other_moves + 1) 

    def stableDiscsFromCorner(self, player):
        
        corner = [0, 7, 56, 63]
               
        for c in corner:
            inc_v, inc_h, horizStop, vertStop = 8, 1, 7, 56
            down, right = False, False
            flat_board = np.array(self.board).flatten()
            if c == 0:
                down = True
                right = True
            
            elif c == 7:
                down = True
                right = False
            
            elif c == 56:
                down = False
                right = True
            
            else:
                down = False
                right = False

            if not right:
                inc_h *= -1
                horizStop *= -1
            
            if not down:
                inc_v *= -1
                vertStop *= -1            
            i = int(c)
            j = int(i)
            while True:
                if i == c+inc_h+horizStop:break
                
                if flat_board[i] == player:
                    while True:
                        if j == i+vertStop:break
                        if flat_board[j] == player and j not in self.POS:
                            self.POS.append(j)
                        else:break
                        j += inc_v
                else:break
                i += inc_h
        return len(self.POS)

    def stability(self, player): #穩定性
        self.POS = []
        my_stables = self.stableDiscsFromCorner(player)
        other_stables = self.stableDiscsFromCorner(2 if player == 1 else 1)
        #print(my_stables, other_stables)
        return 100*(my_stables - other_stables)/(my_stables + other_stables +1)


class ReversiAI(Reversi):
    MIN, MAX = -999, 999
    def __init__(self, board, h, w):
        super().__init__(h, w)
        self.board = board

    def is_corner(self, x, y):
        return x in {0, self.width - 1} and y in {0, self.height - 1}

    def get_AB_AI_move(self, computer): 
          
        if computer == 1:
            #score, move = self.max_alpha_beta(self.MIN, self.MAX, 5, computer)
            score, move = self.max_alpha_beta(self.MIN, self.MAX, 8, computer)
        else:
            #score, move = self.min_alpha_beta(self.MIN, self.MAX, 5, computer) 
            score, move = self.min_alpha_beta(self.MIN, self.MAX, 8, computer) 
        #score, move = self.max_alpha_beta(self.MIN, self.MAX, 550, computer)
        print(score, move)
        return move
    
    def get_AB_AI2_move(self, computer): 
        flat = list(self.board.flatten())
        if flat.count(1) + flat.count(2) <= 45:move = self.get_AI_move(computer)
        else:
            if computer == 1:
                score, move = self.max_alpha_beta(self.MIN, self.MAX, 800, computer)
            else:
                score, move = self.min_alpha_beta(self.MIN, self.MAX, 800, computer)
        
        #print(move)
        return move

    def get_AI_move(self, computer):
        possible_moves = self.get_valid_moves(computer)
        np.random.shuffle(possible_moves)
        #print(possible_moves)
        if possible_moves:
            for x, y in possible_moves:
                if self.is_corner(x, y):
                    return [x, y]

            best_score, best_move = -999, None
            for x, y in possible_moves:
                flips = self.chess(computer, x, y)
                scores = self.evaluate(computer)#self.get_num()[computer] 
                #score = self.get_num()[computer]
                if scores > best_score:
                    best_score, best_move = scores, [x, y]

                self.board[x, y] = 0
                otherplayer = 1 if computer == 2 else 2
                for x, y in flips:
                    self.board[x, y] = otherplayer
            return best_move

    def get_AI_move_num(self, computer):
        possible_moves = self.get_valid_moves(computer)
        np.random.shuffle(possible_moves)
        #print(possible_moves)
        if possible_moves:
            for x, y in possible_moves:
                if self.is_corner(x, y):
                    return [x, y]

            best_score, best_move = -999, None
            for x, y in possible_moves:
                flips = self.chess(computer, x, y)
                scores = self.get_num()[computer] 
                #score = self.get_num()[computer]
                if scores > best_score:
                    best_score, best_move = scores, [x, y]

                self.board[x, y] = 0
                otherplayer = 1 if computer == 2 else 2
                for x, y in flips:
                    self.board[x, y] = otherplayer
            return best_move


    def min_alpha_beta(self, alpha, beta, depth, player):
        best_score, best_move = beta, [-1, -1]
        other_player = 2 if player == 1 else 1
        if depth <= 0:            
            if self.height == 8:best_score = self.evaluate(player)
            elif self.height == 10:best_score = self.get_weighted_score(player)
            else:best_score = self.get_num()[player]            
            return best_score, best_move
        
        elif not self.get_valid_moves(player):            
            best_score, best_move = self.max_alpha_beta(alpha, beta, depth-1, 2 if player == 1 else 1)
            return best_score, [-1, -1]
        
        m = beta
        moves = self.get_valid_moves(player)
        np.random.shuffle(moves)
        for move in moves:
            flips = self.chess(player, move[0], move[1])
            score, best_move = self.max_alpha_beta(alpha, m, depth-1, 2 if player == 1 else 1)

            #restore
            self.board[move[0]][move[1]] = 0
            for x, y in flips:
                self.board[x][y] = other_player

            if score < m:
                m = score
                best_move = move[:]
                best_score = m
            if m <= alpha:
                return best_score, best_move
            return best_score, best_move

    def max_alpha_beta(self, alpha, beta, depth, player):
        best_score, best_move = alpha, [-1, -1]
        #到root
        if depth <= 0:
            if self.height == 8:best_score = self.evaluate(player)
            elif self.height == 10:best_score = self.get_weighted_score(player)
            else:best_score = self.get_num()[player] 
            return best_score, best_move
        
        elif not self.get_valid_moves(player):            
            best_score, best_move = self.min_alpha_beta(alpha, beta, depth-1, 2 if player == 1 else 1)
            return best_score, [-1, -1]
        
        m = alpha
        moves = self.get_valid_moves(player)
        np.random.shuffle(moves)
        for move in moves:
            flips = self.chess(player, move[0], move[1])
            score, best_move = self.min_alpha_beta(m, beta, depth-1, 2 if player == 1 else 1)

            #restore
            self.board[move[0]][move[1]] = 0
            other_player = 2 if player == 1 else 1
            for x, y in flips:
                self.board[x][y] = other_player

            if score > m:
                m = score
                best_move = move[:]
                best_score = m
            if m >= beta:
                return best_score, best_move
            return best_score, best_move

    def miniMax_alpha_beta(self, layers, branch, depth, index, nodes, isPlayer, beta, alpha):
            
        if depth == layers:return nodes[index]
        
        #AI playing:choose the big one
        if isPlayer:
            best = MIN
            for i in range(0, branch):            
                val = self.miniMax_alpha_beta(layers, branch ,depth+1, index*branch+i, nodes, False, beta, alpha) #get player state
                #print(nodes)
                best = max(best, val)
                alpha = max(alpha, best)  
                if beta <= alpha:break
            return best          
        else:
        #player playing:choose the small one
            best = MAX
            for i in range(0, branch):
                val = self.miniMax_alpha_beta(layers, branch ,depth+1, index*branch+i, nodes, True, beta, alpha) #get AI state
                #print(nodes)
                best = min(best, val)
                beta = min(beta, best)  
                if beta <= alpha:break
            return best         
        
class Game(Reversi):
    MIN, MAX = -999, 999
    def __init__(self, h, w):
        super().__init__(h, w)
        self.turn = 1
        self.ai = ReversiAI(self.board, self.height, self.width)

    def get_player_moves(self, player): #使用者輸入棋步
        DIGITS = [str(i) for i in range(1, 10)]
        x, y = [-1, -1]
        while True:
            if not self.get_valid_moves(player):break
            move = input('\rplease input p{player} pos ex. 6 5：').split()            
            if len(move) == 2 and move[0] in DIGITS and move[1] in DIGITS:
                x = int(move[0]) - 1
                y = int(move[1]) - 1
                #print(x, y, self.check_move_valid(x, y, player))
                if self.check_move_valid(x, y, player):break
            print('invalid move, please try again.')
        return [x, y]    

    def show_scores(self, player, computer):#顯示最後棋子比數
        scores = self.get_num()
        print('%10s：%10s\n%10d：%10d'%('Player', 'Computer', scores[player], scores[computer]))
        if scores[player] > scores[computer]:print('Player Win')
        elif scores[player] < scores[computer]:print('Computer Win')
        else:print('Draw')

    def draw_board(self): #畫出棋盤
        h_line =  ' ' * 3 + '+---' * self.width  + '+'
        v_line = (' ' * 3 +'|') * (self.width +1)
        title = '     1'
        for i in range(1,self.width):
            title += ' ' * 3 +str(i+1)
        print(title)
        print(h_line)
        for y in range(self.height):
            #print(v_line)
            print(y+1, end='  ')
            for x in range(self.width):  
                if int(self.board[x][y]) < 1:print(f'|  ', end=' ')
                else:print(f'| {"B" if int(self.board[x][y])==1 else "W"}', end=' ')
            print('|')
            #print(v_line)
            print(h_line)

    def game_loop(self):
        player = 1
        computer = 2 
        cx1, cy1, cx2, cy2 = 0, 0, 0, 0
        mode = input('do u want com v.s. com？')
        self.init_board()
        if mode.lower() == 'y':
            #讓AI互打
            while True:
                if (not self.get_valid_moves(player)) and (not self.get_valid_moves(computer)):break
                pos = self.ai.get_AI_move(player)
                if not pos:break
                cx1, cy1 = list(pos)                
                if cx1 < 0 and not self.get_valid_moves(player):break
                self.chess(player, cx1, cy1)
                self.draw_board()
                
                if (not cx2 and not cy2) and cx2 < 0 and not self.get_valid_moves(computer):break
                pos = self.ai.get_AB_AI_move(computer)
                if not pos:break
                cx2, cy2 = list(pos)      
                self.chess(computer, cx2, cy2)
                self.draw_board()
                #print(self.get_num())
            p1 = self.ai.get_AI_move(player)
            p2 = self.ai.get_AI_move(computer)
            while p1 or p2:
                p1 = self.ai.get_AI_move(player)
                p2 = self.ai.get_AI_move(computer)
                print(p1, p2)
                if p1:self.chess(player, p1[0], p1[1])
                if p2:self.chess(computer, p2[0], p2[1])
            self.draw_board()
            self.show_scores(player, computer)                
        else:
            #其中有一方為機器，可選擇先後攻
            player = eval(input('\rinput player you want to be(1 or 2):')) 
            computer = 2 if player == 1 else 1
            self.turn = 1 if player == 1 else 2
            print(f'\rPlayer {"B" if player == 1 else "W"}|computer {"B" if computer == 1  else "W"}')
            self.init_board()
            while True:
                if not self.get_valid_moves(player) and not self.get_valid_moves(computer):break
                if self.turn == 1:        
                    self.draw_board()
                    px, py = self.get_player_moves(player)
                    #if px < 0:break
                    self.chess(player, px, py)
                    #cx, cy = self.ai.get_AB_AI_move(computer)
                    #if cx < 0:break
                    #cx, cy = self.ai.get_AI_move(computer)
                    pos = self.ai.get_AB_AI2_move(computer)
                    #if not pos:break
                    cx, cy = list(pos)                
                    if cx < 0 and not self.get_valid_moves(computer):break
                    print(cx+1, cy+1)
                    self.chess(computer, cx, cy)
                    print(self.get_num()[player], '：', self.get_num()[computer])
                    
                    
                else:
                    #cx, cy = self.ai.get_AI_move(computer)
                    pos = self.ai.get_AB_AI2_move(computer)
                    if not pos:break
                    cx, cy = list(pos)                
                    #if cx < 0 and not self.get_valid_moves(computer):break
                    print(cx+1, cy+1)
                    self.chess(computer, cx, cy)
                    self.draw_board()
                    px, py = self.get_player_moves(player)
                    if px < 0:break
                    self.chess(player, px, py)
                    print(self.get_num()[player], '：', self.get_num()[computer])
                    
            self.draw_board()
            self.show_scores(player, computer)
        '''
        while True:
            self.init_board()
            player, computer = 
        '''
'''
def input_position(board, player):
    inp = input('please input p{player} pos：ex. 6 5').split()
    row, col = (int(i) for i in inp)
    while board[row, col] != 0:
        print('this pos is already placed')
        inp = input('please input p{player} pos：ex. 6 5').split()
        row, col = (int(i) for i in inp)
    return row, col    

def p1_move(board, row, col):
    board[row, col] = 1
    return board

def p2_move(board, row, col):
    board[row, col] = 2
    return board    
'''

if __name__ == '__main__':
    '''
    board = init_board(8, 8)
    while True:
        p1 = input("P1：").split()
        if p1=='end':break
        board[int(p1[0]), int(p1[1])] = 1
        print(board)

        p2 = input("P2：").split()
        board[int(p2[0]), int(p2[1])] = 2
        print(board)
    '''
    size = eval(input('input board size(4~10)：'))
    game = Game(size, size)
    game.game_loop()
    