import random
import math
import time
from itertools import product
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import List, Tuple, Optional, Any

# ----------------------
# MCTS Node Definition
# ----------------------
class MCTSNode:
    # All piece feature tuples for consistent encoding/decoding
    pieces: List[Tuple[int,int,int,int]] = [(i,j,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

    @staticmethod
    @lru_cache(maxsize=8192)
    def _evaluate_state(board_key: Tuple[Tuple[int,...],...], avail_key: Tuple[Tuple[int,int,int,int],...]) -> float:
        board = [list(row) for row in board_key]
        score = 0.0
        # center control
        for r, c in [(1,1),(1,2),(2,1),(2,2)]:
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
        for r, c in product(range(4), range(4)):
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
        # heuristic evaluation key
        self.heuristic = MCTSNode._evaluate_state(
            tuple(tuple(r) for r in self.board),
            tuple(self.available_pieces)
        )
        # shuffle actions to avoid bias
        self.untried_actions = self._prioritize_actions()

    def _prioritize_actions(self) -> List[Any]:
        actions = self._get_actions()
        random.shuffle(actions)
        return actions

    def _get_actions(self) -> List[Any]:
        if self.is_terminal():
            return []
        if self.player_phase == 'select':
            return list(self.available_pieces)
        return [(r, c) for r, c in product(range(4), range(4)) if self.board[r][c] == 0]

    def is_terminal(self) -> bool:
        if MCTSNode._check_win(self.board):
            return True
        if not self.available_pieces:
            return all(cell != 0 for row in self.board for cell in row)
        return False

    def uct_score(self, total: int, exploration: float, heuristic_weight: float) -> float:
        if self.visits == 0:
            return float('inf')
        return (
            self.wins / self.visits
            + exploration * math.sqrt(math.log(total) / self.visits)
            + heuristic_weight * self.heuristic
        )

    def expand(self) -> 'MCTSNode':
        action = self.untried_actions.pop(0)
        nb, na, np_, sp = self._apply_action(action)
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
        max_steps = 100
        for _ in range(max_steps):
            if MCTSNode._check_win(b):
                return 1.0 if phase == 'place' else 0.0
            if not avail or all(cell != 0 for row in b for cell in row):
                return 0.5
            board_key = tuple(tuple(r) for r in b)
            if phase == 'select':
                safe = [p for p in avail if not MCTSNode._opponent_can_win_cached(board_key, p)]
                choice = random.choice(safe) if safe else random.choice(avail)
                avail.remove(choice)
                selected = choice
                phase = 'place'
            else:
                empties = [(r, c) for r, c in product(range(4), range(4)) if b[r][c] == 0]
                r, c = random.choice(empties)
                b[r][c] = MCTSNode._encode_piece(selected)
                phase = 'select'
        return 0.5

    def _apply_action(self, action: Any) -> Tuple[List[List[int]], List[Tuple[int,int,int,int]], str, Optional[Tuple[int,int,int,int]]]:
        b = [row.copy() for row in self.board]
        a = self.available_pieces.copy()
        if self.player_phase == 'select':
            a.remove(action)
            return b, a, 'place', action
        r, c = action
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
                if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)):
                    return True
        for r in range(3):
            for c in range(3):
                block = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in block:
                    traits = [MCTSNode._decode_piece(cell) for cell in block]
                    if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)):
                        return True
        return False

    @staticmethod
    def _encode_piece(piece: Tuple[int,int,int,int]) -> int:
        return MCTSNode.pieces.index(piece) + 1

    @staticmethod
    def _decode_piece(val: int) -> Tuple[int,int,int,int]:
        return MCTSNode.pieces[val-1]

# -----------------------------
# P1 Player with Enhanced Time Management and Safety
# -----------------------------
class P1:
    MAX_TURN_TIME = 50
    ITERATION_CAP = 2000

    def __init__(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]]):
        self.exploration = 1.4
        self.heuristic_weight = 0.5
        self.root = MCTSNode(board, available_pieces, player_phase='select')
        self.debug = True

    def _search(self, time_limit: float):
        end_time = time.time() + time_limit * 0.9
        iterations = 0
        start = time.time()
        futures = []
        if self.debug:
            print(f"[DEBUG] Search start: limit={time_limit:.2f}s, cap={P1.ITERATION_CAP}")
        with ThreadPoolExecutor(max_workers=4) as executor:
            while time.time() < end_time and iterations < P1.ITERATION_CAP:
                futures.append(executor.submit(self._iterate))
                iterations += 1
                if self.debug and iterations % 100 == 0:
                    print(f"[DEBUG] {iterations} iterations, {time.time()-start:.2f}s elapsed")
            done, _ = wait(futures, timeout=time_limit*0.1, return_when=ALL_COMPLETED)
            if self.debug:
                print(f"[DEBUG] Completed {len(done)}/{len(futures)} tasks")
        if self.debug:
            print(f"[DEBUG] Search end: total iterations={iterations}")

    def _iterate(self):
        node = self.root
        while not node.untried_actions and node.children:
            node = node.best_child(self.exploration, self.heuristic_weight)
        if node.untried_actions:
            node = node.expand()
        result = node.simulate()
        node.backpropagate(result)

    def select_piece(self, time_limit: float = 1.0) -> Tuple[int,int,int,int]:
        t = min(time_limit, P1.MAX_TURN_TIME)
        self._search(t)
        if not self.root.children:
            return random.choice(self.root.available_pieces)
        best = max(self.root.children, key=lambda c: c.visits)
        # re-root to preserve subtree
        best.parent = None
        self.root = best
        return best.selected_piece

    def place_piece(self, piece: Tuple[int,int,int,int], time_limit: float = 1.0) -> Tuple[int,int]:
        # immediate win check
        for r, c in product(range(4), range(4)):
            if self.root.board[r][c] == 0:
                b2 = [row.copy() for row in self.root.board]
                b2[r][c] = MCTSNode._encode_piece(piece)
                if MCTSNode._check_win(b2):
                    return r, c
        # reuse matching child
        for child in self.root.children:
            if child.selected_piece == piece:
                child.parent = None
                self.root = child
                break
        else:
            self.root = MCTSNode(self.root.board, self.root.available_pieces, 'place', selected_piece=piece)
        t = min(time_limit, P1.MAX_TURN_TIME)
        self._search(t)
        if not self.root.children:
            empties = [(r,c) for r,c in product(range(4), range(4)) if self.root.board[r][c]==0]
            return empties[0] if empties else (0,0)
        best = max(self.root.children, key=lambda c: c.visits)
        for r, c in product(range(4), range(4)):
            if self.root.board[r][c] == 0 and best.board[r][c] != 0:
                return r, c
        empties = [(r,c) for r,c in product(range(4), range(4)) if self.root.board[r][c]==0]
        return empties[0] if empties else (0,0)

    def record_game(self, won: bool):
        pass
