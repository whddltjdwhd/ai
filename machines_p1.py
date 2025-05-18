import random
import math
import time
from itertools import product
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
from typing import List, Tuple, Optional, Any

# ----------------------
# MCTS Node Definition
# ----------------------
class MCTSNode:
    # All 16 pieces as tuples of features
    pieces: List[Tuple[int,int,int,int]] = [
        (i,j,k,l)
        for i in range(2)
        for j in range(2)
        for k in range(2)
        for l in range(2)
    ]

    @staticmethod
    @lru_cache(maxsize=8192)
    def _evaluate_state(board_key: Tuple[Tuple[int,...],...], avail_key: Tuple[Tuple[int,int,int,int],...]) -> float:
        board = [list(row) for row in board_key]
        score = 0.0
        # center control
        for r,c in [(1,1),(1,2),(2,1),(2,2)]:
            if board[r][c] == 0:
                score += 0.1
        # 3-in-line potential
        for line in MCTSNode._all_lines(board):
            filled = [cell for cell in line if cell]
            if len(filled) >= 2:
                traits = [MCTSNode._decode_piece(cell) for cell in filled]
                if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)):
                    score += 0.2
        return score

    @staticmethod
    @lru_cache(maxsize=16384)
    def _opponent_can_win_cached(board_key: Tuple[Tuple[int,...],...], piece: Tuple[int,int,int,int]) -> bool:
        board = [list(row) for row in board_key]
        for r,c in product(range(4), range(4)):
            if board[r][c] == 0:
                b2 = [row.copy() for row in board]
                b2[r][c] = MCTSNode._encode_piece(piece)
                if MCTSNode._check_win(b2):
                    return True
        return False

    def __init__(
        self,
        board: List[List[int]],
        available_pieces: List[Tuple[int,int,int,int]],
        player_phase: str,
        selected_piece: Optional[Tuple[int,int,int,int]] = None,
        parent: Optional['MCTSNode'] = None
    ):
        self.board = [row.copy() for row in board]
        self.available_pieces = available_pieces.copy()
        self.player_phase = player_phase
        self.selected_piece = selected_piece
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.wins = 0.0
        self.visits = 0
        # heuristic
        self.heuristic = MCTSNode._evaluate_state(
            tuple(tuple(r) for r in self.board), tuple(self.available_pieces)
        )
        # prioritized actions
        actions = self._get_actions()
        random.shuffle(actions)
        self.untried_actions = actions

    def _get_actions(self) -> List[Any]:
        if self.is_terminal(): return []
        if self.player_phase == 'select': return list(self.available_pieces)
        return [(r,c) for r,c in product(range(4), range(4)) if self.board[r][c]==0]

    def is_terminal(self) -> bool:
        if MCTSNode._check_win(self.board): return True
        if not self.available_pieces:
            return all(cell!=0 for row in self.board for cell in row)
        return False

    def uct_score(self, total: int, exploration: float, heuristic_weight: float) -> float:
        if self.visits==0: return float('inf')
        return (
            self.wins/self.visits
            + exploration*math.sqrt(math.log(total)/self.visits)
            + heuristic_weight*self.heuristic
        )

    def expand(self) -> 'MCTSNode':
        action = self.untried_actions.pop(0)
        nb,na,np_,sp = self._apply_action(action)
        child = MCTSNode(nb, na, np_, selected_piece=sp, parent=self)
        self.children.append(child)
        return child

    def best_child(self, exploration: float, heuristic_weight: float) -> 'MCTSNode':
        total = sum(c.visits for c in self.children) or 1
        return max(
            self.children,
            key=lambda c: c.uct_score(total, exploration, heuristic_weight)
        )

    def backpropagate(self, result: float):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def simulate(self) -> float:
        b = [row.copy() for row in self.board]
        avail = self.available_pieces.copy()
        phase = self.player_phase
        selected = self.selected_piece
        for _ in range(100):
            # immediate win
            for p in avail:
                for r,c in product(range(4), range(4)):
                    if b[r][c]==0:
                        b2=[row.copy() for row in b]
                        b2[r][c]=MCTSNode._encode_piece(p)
                        if MCTSNode._check_win(b2):
                            if self.parent and self.parent.parent is None:
                                print(f"[DEBUG] Simulation immediate win detected for piece {p} at {(r,c)}")
                            return 1.0 if phase=='place' else 0.0
            if MCTSNode._check_win(b):
                return 1.0 if phase=='place' else 0.0
            if not avail:
                return 0.5
            board_key = tuple(tuple(r) for r in b)
            if phase=='select':
                safe=[p for p in avail if not MCTSNode._opponent_can_win_cached(board_key,p)]
                if safe:
                    weights=[1+0.5*sum(1 for (cr,cc) in [(1,1),(1,2),(2,1),(2,2)] if b[cr][cc]==0) for p in safe]
                    choice=random.choices(safe,weights=weights,k=1)[0]
                else:
                    choice=random.choice(avail)
                if self.parent and self.parent.parent is None:
                    print(f"[DEBUG] Simulation select phase choice={choice}")
                avail.remove(choice); selected=choice; phase='place'
            else:
                empties=[(r,c) for r,c in product(range(4), range(4)) if b[r][c]==0]
                r,c=random.choice(empties)
                if self.parent and self.parent.parent is None:
                    print(f"[DEBUG] Simulation place phase at {(r,c)} with piece {selected}")
                b[r][c]=MCTSNode._encode_piece(selected); phase='select'
        return 0.5

    def _apply_action(self, action: Any) -> Tuple[List[List[int]],List[Tuple[int,int,int,int]],str,Optional[Tuple[int,int,int,int]]]:
        b=[row.copy() for row in self.board]; a=self.available_pieces.copy()
        if self.player_phase=='select':
            a.remove(action);
            print(f"[DEBUG] expand: selecting piece {action}")
            return b,a,'place',action
        r,c=action;
        print(f"[DEBUG] expand: placing piece {self.selected_piece} at {(r,c)}")
        b[r][c]=MCTSNode._encode_piece(self.selected_piece)
        return b,a,'select',None

    @staticmethod
    def _all_lines(board: List[List[int]])->List[List[int]]:
        L=[]
        for i in range(4):
            L.append([board[i][j] for j in range(4)]); L.append([board[j][i] for j in range(4)])
        L.append([board[i][i] for i in range(4)]); L.append([board[i][3-i] for i in range(4)])
        return L

    @staticmethod
    def _check_win(board: List[List[int]])->bool:
        for line in MCTSNode._all_lines(board):
            if 0 not in line:
                traits=[MCTSNode._decode_piece(c) for c in line]
                if any(all(t[i]==traits[0][i] for t in traits) for i in range(4)): return True
        for r in range(3):
            for c in range(3):
                blk=[board[r][c],board[r][c+1],board[r+1][c],board[r+1][c+1]]
                if 0 not in blk:
                    traits=[MCTSNode._decode_piece(c) for c in blk]
                    if any(all(t[i]==traits[0][i] for t in traits) for i in range(4)): return True
        return False

    @staticmethod
    def _encode_piece(piece: Tuple[int,int,int,int]) -> int:
        return MCTSNode.pieces.index(piece)+1

    @staticmethod
    def _decode_piece(val: int) -> Tuple[int,int,int,int]:
        return MCTSNode.pieces[val-1]

# -----------------------------
# P1 Player with Dynamic and Safe MCTS
# -----------------------------
class P1:
    MAX_TURN_TIME=10
    ITERATION_CAP=1000

    def __init__(self,board:List[List[int]],available_pieces:List[Tuple[int,int,int,int]]):
        self.exploration_base=1.4;self.heuristic_weight_base=0.5
        self._adjust_parameters(board)
        self.root=MCTSNode(board,available_pieces,'select')
        self.debug=True

    def _adjust_parameters(self,board):
        empty=sum(1 for r in board for c in r if c==0)
        prog=1-empty/16
        self.exploration=self.exploration_base*(1-0.3*prog)
        self.heuristic_weight=self.heuristic_weight_base*(1+0.5*prog)

    def _search(self,tl):
        end=time.time()+tl*0.9;it=0;start=time.time()
        print(f"[DEBUG] Search start: limit={tl:.2f}, exploration={self.exploration:.2f}, heuristic_weight={self.heuristic_weight:.2f}")
        with ThreadPoolExecutor(max_workers=4) as ex:
            while time.time()<end and it<P1.ITERATION_CAP:
                batch=[ex.submit(self._iterate) for _ in range(min(20,P1.ITERATION_CAP-it))]
                done,_=wait(batch,return_when=ALL_COMPLETED)
                for f in done:
                    f.result();it+=1
                if it%100==0:
                    print(f"[DEBUG] {it} iterations, elapsed={time.time()-start:.2f}s")
        print(f"[DEBUG] Search end: total iterations={it}")

    def _iterate(self):
        node=self.root
        while not node.untried_actions and node.children:
            node=node.best_child(self.exploration,self.heuristic_weight)
        if node.untried_actions:
            node=node.expand()
        res=node.simulate(); node.backpropagate(res)

    def select_piece(self,tl=1.0):
        print(f"[DEBUG] select_piece called, board=\n{self.root.board}")
        key=tuple(tuple(r) for r in self.root.board)
        risks={p:1 if MCTSNode._opponent_can_win_cached(key,p) else 0 for p in self.root.available_pieces}
        t=min(tl,P1.MAX_TURN_TIME); self._search(t)
        print(f"[DEBUG] select_piece post-search, children count={len(self.root.children)}")
        if not self.root.children:
            safe=[p for p,r in risks.items() if r==0]
            choice=random.choice(safe) if safe else random.choice(self.root.available_pieces)
            print(f"[DEBUG] select_piece fallback choice={choice}")
            return choice
        scored=[(c,c.visits*(1-0.8*risks.get(c.selected_piece,0))) for c in self.root.children]
        best,_=max(scored,key=lambda x:x[1]);best.parent=None;self.root=best
        print(f"[DEBUG] select_piece result={best.selected_piece}")
        return best.selected_piece

    def place_piece(self,piece,tl=1.0):
        print(f"[DEBUG] place_piece called with piece={piece}, board=\n{self.root.board}")
        # immediate win
        for r,c in product(range(4),range(4)):
            if self.root.board[r][c]==0:
                b2=[row.copy() for row in self.root.board];b2[r][c]=MCTSNode._encode_piece(piece)
                if MCTSNode._check_win(b2): print(f"[DEBUG] Immediate win at {(r,c)}"); return r,c
        # danger
        danger=[]
        for r,c in product(range(4),range(4)):
            if self.root.board[r][c]==0:
                b2=[row.copy() for row in self.root.board]; b2[r][c]=MCTSNode._encode_piece(piece)
                key2=tuple(tuple(r) for r in b2)
                if any(MCTSNode._opponent_can_win_cached(key2,p) for p in self.root.available_pieces if p!=piece): danger.append((r,c))
        print(f"[DEBUG] Dangerous positions: {danger}")
        # reuse subtree
        for ch in self.root.children:
            if ch.selected_piece==piece: ch.parent=None; self.root=ch; print(f"[DEBUG] Reused child for piece placement") ; break
        else:
            self.root=MCTSNode(self.root.board,self.root.available_pieces,'place',piece)
            print(f"[DEBUG] Created new root for place phase")
        t=min(tl,P1.MAX_TURN_TIME); self._search(t)
        print(f"[DEBUG] place_piece post-search, children count={len(self.root.children)}")
        if not self.root.children:
            empt=[(r,c) for r,c in product(range(4),range(4)) if self.root.board[r][c]==0]
            safe=[pos for pos in empt if pos not in danger]
            choice=random.choice(safe) if safe else (empt[0] if empt else (0,0))
            print(f"[DEBUG] place_piece fallback choice={choice}")
            return choice
        best=max(self.root.children,key=lambda c:c.visits)
        for r,c in product(range(4),range(4)):
            if self.root.board[r][c]==0 and best.board[r][c]!=0:
                if (r,c) in danger and len(self.root.children)>1:
                    sec=sorted(self.root.children,key=lambda c:c.visits)[-2]
                    for r2,c2 in product(range(4),repeat=2):
                        if self.root.board[r2][c2]==0 and sec.board[r2][c2]!=0 and (r2,c2) not in danger:
                            print(f"[DEBUG] place_piece selected second best at {(r2,c2)}")
                            return r2,c2
                print(f"[DEBUG] place_piece selected best at {(r,c)}")
                return r,c
        empt=[(r,c) for r,c in product(range(4),range(4)) if self.root.board[r][c]==0]
        choice=empt[0] if empt else (0,0)
        print(f"[DEBUG] place_piece final fallback choice={choice}")
        return choice

    def record_game(self,won:bool):
        print(f"[DEBUG] Game result recorded: won={won}")
        pass