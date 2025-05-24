import random
import math
import time
from itertools import product
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import List, Tuple, Optional, Any


class MCTSNode:
    """
    Node in a Monte Carlo Tree Search representing a game state.
    """
    # Precompute all 16 possible piece feature tuples
    pieces: List[Tuple[int,int,int,int]] = [
        (i, j, k, l)
        for i in range(2)
        for j in range(2)
        for k in range(2)
        for l in range(2)
    ]

    @staticmethod
    @lru_cache(maxsize=8192)
    def _evaluate_state(
        board_key: Tuple[Tuple[int, ...], ...],
        avail_key: Tuple[Tuple[int, int, int, int], ...]
    ) -> float:
        """
        Heuristic evaluation of board state for non-terminal nodes.
        Rewards central control and pattern potential.
        """
        board = [list(row) for row in board_key]
        score = 0.0
        # central squares bonus
        for r, c in [(1,1),(1,2),(2,1),(2,2)]:
            if board[r][c] == 0:
                score += 0.1
        # potential lines of 3
        for line in MCTSNode._all_lines(board):
            filled = [cell for cell in line if cell]
            if len(filled) >= 2:
                traits = [MCTSNode._decode_piece(cell) for cell in filled]
                if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)):
                    score += 0.2
        return score

    @staticmethod
    @lru_cache(maxsize=16384)
    def _opponent_can_win_cached(
        board_key: Tuple[Tuple[int, ...], ...],
        piece: Tuple[int,int,int,int]
    ) -> bool:
        """
        Fast check if opponent can win by placing `piece` anywhere.
        """
        board = [list(row) for row in board_key]
        for r, c in product(range(4), range(4)):
            if board[r][c] == 0:
                board_copy = [row.copy() for row in board]
                board_copy[r][c] = MCTSNode._encode_piece(piece)
                if MCTSNode._check_win(board_copy):
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
        self.player_phase = player_phase  # 'select' or 'place'
        self.selected_piece = selected_piece
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.wins = 0.0
        self.visits = 0
        # heuristic value cached by board and avail keys
        self.heuristic = MCTSNode._evaluate_state(
            tuple(tuple(r) for r in self.board),
            tuple(self.available_pieces)
        )
        actions = self._get_actions()
        random.shuffle(actions)
        self.untried_actions = actions

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

    def uct_score(self, total_visits: int, exploration: float, heuristic_weight: float) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(total_visits) / self.visits)
        return exploitation + exploration_term + heuristic_weight * self.heuristic

    def expand(self) -> 'MCTSNode':
        action = self.untried_actions.pop(0)
        next_board, next_avail, next_phase, next_piece = self._apply_action(action)
        child = MCTSNode(next_board, next_avail, next_phase, selected_piece=next_piece, parent=self)
        self.children.append(child)
        return child

    def best_child(self, exploration: float, heuristic_weight: float) -> 'MCTSNode':
        total = sum(child.visits for child in self.children) or 1
        return max(self.children, key=lambda c: c.uct_score(total, exploration, heuristic_weight))

    def backpropagate(self, result: float):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def simulate(self) -> float:
        board_state = [row.copy() for row in self.board]
        pieces_left = self.available_pieces.copy()
        phase = self.player_phase
        current_piece = self.selected_piece

        for _ in range(200):
            # immediate win check
            for p in pieces_left:
                for r, c in product(range(4), range(4)):
                    if board_state[r][c] == 0:
                        trial = [row.copy() for row in board_state]
                        trial[r][c] = MCTSNode._encode_piece(p)
                        if MCTSNode._check_win(trial):
                            return 1.0 if phase == 'place' else 0.0
            # terminal board win check
            if MCTSNode._check_win(board_state):
                return 1.0 if phase == 'place' else 0.0
            if not pieces_left:
                return 0.5  # draw

            if phase == 'select':
                safe_pieces = [p for p in pieces_left if not MCTSNode._opponent_can_win_cached(tuple(tuple(r) for r in board_state), p)]
                choices = safe_pieces or pieces_left
                weights = [1.0 + 0 for _ in choices]
                choice = random.choices(choices, weights, k=1)[0]
                pieces_left.remove(choice)
                current_piece = choice
                phase = 'place'
            else:
                empties = [(r, c) for r, c in product(range(4), range(4)) if board_state[r][c] == 0]
                r, c = random.choice(empties)
                board_state[r][c] = MCTSNode._encode_piece(current_piece)
                phase = 'select'

        return 0.5

    def _apply_action(self, action: Any) -> Tuple[List[List[int]], List[Tuple[int,int,int,int]], str, Optional[Tuple[int,int,int,int]]]:
        new_board = [row.copy() for row in self.board]
        new_avail = self.available_pieces.copy()
        if self.player_phase == 'select':
            new_avail.remove(action)
            return new_board, new_avail, 'place', action
        r, c = action
        new_board[r][c] = MCTSNode._encode_piece(self.selected_piece)
        return new_board, new_avail, 'select', None

    @staticmethod
    def _all_lines(board: List[List[int]]) -> List[List[int]]:
        lines: List[List[int]] = []
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
                traits = [MCTSNode._decode_piece(v) for v in line]
                if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)): return True
        for r in range(3):
            for c in range(3):
                block = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in block:
                    traits = [MCTSNode._decode_piece(v) for v in block]
                    if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)): return True
        return False

    @staticmethod
    def _encode_piece(piece: Tuple[int,int,int,int]) -> int:
        return MCTSNode.pieces.index(piece) + 1

    @staticmethod
    def _decode_piece(val: int) -> Tuple[int,int,int,int]:
        return MCTSNode.pieces[val-1]


class P1:
    """
    MCTS-based player with enhanced strategy feedback applied.
    """
    MAX_TURN_TIME = 10
    ITERATION_CAP = 2300

    def __init__(
        self,
        board: List[List[int]],
        available_pieces: List[Tuple[int,int,int,int]]
    ):
        # Determine first/second player
        is_first = len(available_pieces) % 2 == 0
        self.exploration_base = 1.4 if is_first else 1.6
        self.heuristic_weight_base = 0.35 if is_first else 0.25
        # Adjust for game stage
        self._adjust_parameters(board)
        self.root = MCTSNode(board, available_pieces, 'select')
        self.last_search_iterations = 0

    def _adjust_parameters(self, board: List[List[int]]):
        empty = sum(cell == 0 for row in board for cell in row)
        if empty >= 12:
            self.exploration = self.exploration_base * 1.2
            self.heuristic_weight = self.heuristic_weight_base * 0.8
        elif empty >= 6:
            self.exploration = self.exploration_base
            self.heuristic_weight = self.heuristic_weight_base * 1.2
        else:
            self.exploration = self.exploration_base * 0.7
            self.heuristic_weight = self.heuristic_weight_base * 1.5

    def _danger_level(self, pos: Tuple[int,int], piece: Tuple[int,int,int,int]) -> int:
        # Evaluate risk if piece placed at pos
        board_copy = [row.copy() for row in self.root.board]
        board_copy[pos[0]][pos[1]] = MCTSNode._encode_piece(piece)
        key = tuple(tuple(r) for r in board_copy)
        for p in self.root.available_pieces:
            if p != piece and MCTSNode._opponent_can_win_cached(key, p):
                return 3
        return 0

    def _iterate(self):
        node = self.root
        # Selection
        while not node.untried_actions and node.children:
            node = node.best_child(self.exploration, self.heuristic_weight)
        # Expansion
        if node.untried_actions:
            node = node.expand()
        # Simulation & Backpropagation
        result = node.simulate()
        node.backpropagate(result)

    def _search(self):
        iterations = 0
        end_time = time.time() + P1.MAX_TURN_TIME * 0.9
        batch = min(20, P1.ITERATION_CAP)
        with ThreadPoolExecutor(max_workers=6) as executor:
            while time.time() < end_time and iterations < P1.ITERATION_CAP:
                remaining = end_time - time.time()
                cb = min(batch, P1.ITERATION_CAP - iterations)
                futures = [executor.submit(self._iterate) for _ in range(cb)]
                done, _ = wait(futures, timeout=min(0.05, remaining), return_when=FIRST_COMPLETED)
                for f in done:
                    try:
                        f.result()
                        iterations += 1
                    except:
                        pass
        self.last_search_iterations = iterations

    def select_piece(self) -> Tuple[int,int,int,int]:
        empty = sum(cell == 0 for row in self.root.board for cell in row)
        # Early game: danger-based selection
        if empty >= 12:
            empties = [(r, c) for r, c in product(range(4), range(4)) if self.root.board[r][c] == 0]
            danger_scores = {p: max(self._danger_level(e, p) for e in empties) for p in self.root.available_pieces}
            return min(self.root.available_pieces, key=lambda p: danger_scores[p])
        # Mid/late game: MCTS
        self._search()
        if not self.root.children:
            return random.choice(self.root.available_pieces)
        best = max(self.root.children, key=lambda c: c.visits)
        self.root = best
        choice = best.selected_piece
        print(f"SELECT_PIECE -> {choice}, iters={self.last_search_iterations}")
        return choice

    def place_piece(self, piece: Tuple[int,int,int,int]) -> Tuple[int,int]:
        # Immediate win if available
        for r, c in product(range(4), range(4)):
            if self.root.board[r][c] == 0:
                temp = [row.copy() for row in self.root.board]
                temp[r][c] = MCTSNode._encode_piece(piece)
                if MCTSNode._check_win(temp):
                    print(f"PLACE_PIECE -> ({r},{c}), iters={self.last_search_iterations}")
                    return (r, c)
        # Identify dangerous spots
        danger_spots = [(r, c) for r, c in product(range(4), range(4)) if self.root.board[r][c]==0 and self._danger_level((r,c), piece)>0]
        # Search
        self._search()
        # No children: pick safe or first empty
        if not self.root.children:
            empties = [(r,c) for r,c in product(range(4), range(4)) if self.root.board[r][c]==0]
            safe = [e for e in empties if e not in danger_spots]
            choice = safe[0] if safe else empties[0]
            print(f"PLACE_PIECE -> {choice}, iters={self.last_search_iterations}")
            return choice
        # Use child visitation to pick move not in danger
        best = max(self.root.children, key=lambda c: c.visits)
        for r, c in product(range(4), range(4)):
            if best.board[r][c]!=self.root.board[r][c] and self.root.board[r][c]==0 and (r,c) not in danger_spots:
                print(f"PLACE_PIECE -> ({r},{c}), iters={self.last_search_iterations}")
                return (r, c)
        # Fallback
        fallback = next((e for e in product(range(4), range(4)) if self.root.board[e[0]][e[1]]==0), (0, 0))
        print(f"PLACE_PIECE -> {fallback}, iters={self.last_search_iterations}")
        return fallback

    def record_game(self, won: bool):
        # Placeholder for recording outcome
        pass
