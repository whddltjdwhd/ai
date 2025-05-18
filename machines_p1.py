import random
import math
import time
from itertools import product
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Any

# ----------------------
# MCTS Node Definition
# ----------------------
class MCTSNode:
    @staticmethod
    @lru_cache(maxsize=1024)
    def _evaluate_state(board_key: Tuple[Tuple[int,...],...], avail_key: Tuple[Tuple[int,int,int,int],...]) -> float:
        board = [list(row) for row in board_key]
        available = list(avail_key)
        score = 0.0
        # Center control heuristic
        for r, c in [(1,1),(1,2),(2,1),(2,2)]:
            if board[r][c] == 0:
                score += 0.1
        # Pattern potential
        for line in MCTSNode._all_lines(board):
            filled = [cell for cell in line if cell]
            if len(filled) >= 2:
                traits = [MCTSNode._decode_piece(cell) for cell in filled]
                for i in range(4):
                    if all(t[i] == traits[0][i] for t in traits):
                        score += 0.2
        # Threat detection
        for p in available:
            if MCTSNode._opponent_can_win_state(board, p):
                score -= 0.3
        return score

    def __init__(self,
                 board: List[List[int]],
                 available_pieces: List[Tuple[int,int,int,int]],
                 player_phase: str,
                 selected_piece: Optional[Tuple[int,int,int,int]] = None,
                 parent: Optional['MCTSNode'] = None):
        self.board = [row.copy() for row in board]
        self.available_pieces = available_pieces.copy()
        self.player_phase = player_phase
        self.selected_piece = selected_piece
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.wins = 0.0
        self.visits = 0
        # compute heuristic
        board_key = tuple(tuple(row) for row in self.board)
        avail_key = tuple(self.available_pieces)
        self.heuristic = MCTSNode._evaluate_state(board_key, avail_key)
        self.untried_actions = self._get_actions()

    def _get_actions(self) -> List[Any]:
        if self.is_terminal():
            return []
        if self.player_phase == 'select':
            return list(self.available_pieces)
        return [(r,c) for r in range(4) for c in range(4) if self.board[r][c] == 0]

    def is_terminal(self) -> bool:
        return self._check_win(self.board) or (not self.available_pieces and all(cell != 0 for row in self.board for cell in row))

    def uct_score(self, total: int, exploration: float, heuristic_weight: float) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(total) / self.visits)
        return exploitation + exploration_term + heuristic_weight * self.heuristic

    def expand(self) -> 'MCTSNode':
        action = self.untried_actions.pop()
        next_board, next_avail, next_phase, next_piece = self._apply_action(action)
        child = MCTSNode(next_board, next_avail, next_phase, selected_piece=next_piece, parent=self)
        self.children.append(child)
        return child

    def best_child(self, exploration: float, heuristic_weight: float) -> 'MCTSNode':
        total_visits = sum(child.visits for child in self.children) or 1
        return max(self.children, key=lambda c: c.uct_score(total_visits, exploration, heuristic_weight))

    def backpropagate(self, result: float):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def simulate(self) -> float:
        b = [row.copy() for row in self.board]
        avail = self.available_pieces.copy()
        phase = self.player_phase
        selected = self.selected_piece  # 로컬 변수로 복사
        step = 0
        max_steps = 100
        
        while step < max_steps:  # 무한 루프 방지
            step += 1
            
            if self._check_win(b):
                return 1.0 if phase == 'place' else 0.0
                
            if not avail or all(b[r][c] != 0 for r in range(4) for c in range(4)):
                return 0.5  # 무승부
                
            if phase == 'select':
                # 안전한 피스 중에서 무작위 선택
                safe = [p for p in avail if not MCTSNode._opponent_can_win_state(b, p)]
                choice = random.choice(safe) if safe else random.choice(avail)
                avail.remove(choice)
                selected = choice  # 로컬 변수 사용
                phase = 'place'
            else:
                # 빈 위치 중에서 무작위 선택
                empties = [(r,c) for r in range(4) for c in range(4) if b[r][c] == 0]
                if not empties:
                    return 0.5  # 무승부
                    
                r, c = random.choice(empties)  # 무작위성 추가
                b[r][c] = MCTSNode._encode_piece(selected)
                
                # 승리 확인
                if self._check_win(b):
                    return 1.0  # place 단계에서는 현재 플레이어 승리
                    
                phase = 'select'
        
        # 최대 단계 도달 시 무승부
        return 0.5
    
    def _apply_action(self, action: Any) -> Tuple[List[List[int]], List[Tuple[int,int,int,int]], str, Optional[Tuple[int,int,int,int]]]:
        b = [row.copy() for row in self.board]
        a = self.available_pieces.copy()
        if self.player_phase == 'select':
            a.remove(action)
            return b, a, 'place', action
        r,c = action
        b[r][c] = MCTSNode._encode_piece(self.selected_piece)
        return b, a, 'select', None

    @staticmethod
    def _all_lines(board: List[List[int]]) -> List[List[int]]:
        lines = []
        for i in range(4):
            lines.append([board[i][j] for j in range(4)])
            lines.append([board[j][i] for j in range(4)])
        lines.append([board[i][i] for i in range(4)])
        lines.append([board[i][3-i] for i in range(4)])
        return lines

    @staticmethod
    def _check_win(board: List[List[int]]) -> bool:
        for line in MCTSNode._all_lines(board):
            if 0 not in line:
                traits = [MCTSNode._decode_piece(cell) for cell in line]
                for i in range(4):
                    if all(t[i] == traits[0][i] for t in traits):
                        return True
        # 2x2 blocks
        for r in range(3):
            for c in range(3):
                block = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in block:
                    traits = [MCTSNode._decode_piece(cell) for cell in block]
                    for i in range(4):
                        if all(t[i] == traits[0][i] for t in traits):
                            return True
        return False

    @staticmethod
    def _opponent_can_win_state(board: List[List[int]], piece: Tuple[int,int,int,int]) -> bool:
        for r,c in product(range(4), range(4)):
            if board[r][c] == 0:
                b2 = [row.copy() for row in board]
                b2[r][c] = MCTSNode._encode_piece(piece)
                if MCTSNode._check_win(b2):
                    return True
        return False

    @staticmethod
    def _encode_piece(piece: Tuple[int,int,int,int]) -> int:
        return (piece[0]<<3)|(piece[1]<<2)|(piece[2]<<1)|piece[3] + 1

    @staticmethod
    def _decode_piece(val: int) -> Tuple[int,int,int,int]:
        v = val-1
        return ((v>>3)&1, (v>>2)&1, (v>>1)&1, v&1)

# -----------------------------
# P1 Player with MCTS
# -----------------------------
class P1:
    def __init__(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]]):
        self.root = MCTSNode(board, available_pieces, player_phase='select')
        self.exploration = 1.4
        self.heuristic_weight = 0.5

    def _search(self, time_limit: float):
        end = time.time() + time_limit
        futures = []
        
        with ThreadPoolExecutor(max_workers=4) as exec:
            # 시간 내에 작업 제출
            while time.time() < end:
                futures.append(exec.submit(self._iterate))
                
            # 최소한 몇 개의 작업은 완료되길 기다림
            wait_end = min(end + 0.1, time.time() + 0.2)
            completed = 0
            
            while time.time() < wait_end and completed < min(10, len(futures)):
                for f in futures:
                    if f.done():
                        completed += 1

    def _iterate(self):
        node = self.root
        while not node.untried_actions and node.children:
            node = node.best_child(self.exploration, self.heuristic_weight)
        if node.untried_actions:
            node = node.expand()
        result = node.simulate()
        node.backpropagate(result)

    def select_piece(self, time_limit: float = 1.0) -> Tuple[int,int,int,int]:
        self._search(time_limit)
        best = max(self.root.children, key=lambda c: c.visits)
        piece = best.selected_piece
        self.root = MCTSNode(best.board, best.available_pieces, 'place', selected_piece=piece)
        return piece

    def place_piece(self, piece: Tuple[int,int,int,int], time_limit: float = 1.0) -> Tuple[int,int]:
        # 피스가 선택되었는지 확인
        self.root = MCTSNode(self.root.board, self.root.available_pieces, 'place', selected_piece=piece)
        
        self._search(time_limit)
        
        # 자식이 없는 경우에 대한 안전장치
        if not self.root.children:
            # 빈 위치 중 첫 번째 반환
            for r in range(4):
                for c in range(4):
                    if self.root.board[r][c] == 0:
                        return r, c
            return 0, 0
        
        best = max(self.root.children, key=lambda c: c.visits)
        
        for r in range(4):
            for c in range(4):
                if self.root.board[r][c] != best.board[r][c]:
                    return r, c
                    
        # 위치를 찾지 못한 경우 안전하게 빈 위치 반환
        for r in range(4):
            for c in range(4):
                if self.root.board[r][c] == 0:
                    return r, c
                    
        return 0, 0