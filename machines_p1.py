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
                if any(all(t[i] == traits[0][i] for t in traits)
                       for i in range(4)):
                    score += 0.2
        return score

    @staticmethod
    @lru_cache(maxsize=16384)
    def _opponent_can_win_cached(
        board_key: Tuple[Tuple[int, ...], ...],
        piece: Tuple[int, int, int, int]
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
        """
        Returns available actions: piece selections or board placements.
        """
        if self.is_terminal():
            return []
        if self.player_phase == 'select':
            return list(self.available_pieces)
        return [
            (r, c)
            for r, c in product(range(4), range(4))
            if self.board[r][c] == 0
        ]

    def is_terminal(self) -> bool:
        """
        Checks for terminal state: winning or full board with no pieces.
        """
        if MCTSNode._check_win(self.board):
            return True
        if not self.available_pieces:
            return all(
                cell != 0
                for row in self.board
                for cell in row
            )
        return False

    def uct_score(
        self,
        total_visits: int,
        exploration: float,
        heuristic_weight: float
    ) -> float:
        """
        Calculates UCT score for child selection.
        """
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(total_visits) / self.visits
        )
        return exploitation + exploration_term + heuristic_weight * self.heuristic

    def expand(self) -> 'MCTSNode':
        """
        Expands one untried action, returns the new child node.
        """
        action = self.untried_actions.pop(0)
        next_board, next_avail, next_phase, next_piece = self._apply_action(action)
        child = MCTSNode(
            next_board,
            next_avail,
            next_phase,
            selected_piece=next_piece,
            parent=self
        )
        self.children.append(child)
        return child

    def best_child(
        self,
        exploration: float,
        heuristic_weight: float
    ) -> 'MCTSNode':
        """
        Selects the best child by UCT score.
        """
        total = sum(child.visits for child in self.children) or 1
        return max(
            self.children,
            key=lambda c: c.uct_score(total, exploration, heuristic_weight)
        )

    def backpropagate(self, result: float):
        """
        Backpropagates simulation result up the tree.
        """
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def simulate(self) -> float:
        """
        Runs a biased random simulation from this node.
        """
        board_state = [row.copy() for row in self.board]
        pieces_left = self.available_pieces.copy()
        phase = self.player_phase
        current_piece = self.selected_piece

        for _ in range(100):
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
                # weighted piece selection
                safe_pieces = [
                    p for p in pieces_left
                    if not MCTSNode._opponent_can_win_cached(
                        tuple(tuple(r) for r in board_state), p
                    )
                ]
                choices = safe_pieces or pieces_left
                weights = [
                    1.0 + self._calculate_piece_value(board_state, p)
                    for p in choices
                ]
                choice = random.choices(choices, weights, k=1)[0]
                pieces_left.remove(choice)
                current_piece = choice
                phase = 'place'
            else:
                # weighted placement
                empties = [
                    (r, c)
                    for r, c in product(range(4), range(4))
                    if board_state[r][c] == 0
                ]
                weights = []
                for (r, c) in empties:
                    w = 1.0
                    if (r, c) in [(1,1),(1,2),(2,1),(2,2)]:
                        w += 0.5
                    if self._can_form_line(board_state, r, c, current_piece):
                        w += 1.0
                    weights.append(w)
                r, c = random.choices(empties, weights, k=1)[0]
                board_state[r][c] = MCTSNode._encode_piece(current_piece)
                phase = 'select'

        return 0.5  # max depth reached

    def _apply_action(
        self,
        action: Any
    ) -> Tuple[List[List[int]], List[Tuple[int,int,int,int]], str, Optional[Tuple[int,int,int,int]]]:
        """
        Applies an action, returning new state and next phase.
        """
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
        """
        Returns all rows, columns, and diagonals of the 4x4 board.
        """
        lines: List[List[int]] = []
        for i in range(4):
            lines.append([board[i][j] for j in range(4)])
            lines.append([board[j][i] for j in range(4)])
        lines.append([board[i][i] for i in range(4)])
        lines.append([board[i][3-i] for i in range(4)])
        return lines

    @staticmethod
    def _check_win(board: List[List[int]]) -> bool:
        """
        Checks for any winning line or 2x2 block.
        """
        # lines
        for line in MCTSNode._all_lines(board):
            if 0 not in line:
                traits = [MCTSNode._decode_piece(v) for v in line]
                if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)):
                    return True
        # 2x2 blocks
        for r in range(3):
            for c in range(3):
                block = [
                    board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]
                ]
                if 0 not in block:
                    traits = [MCTSNode._decode_piece(v) for v in block]
                    if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)):
                        return True
        return False

    @staticmethod
    def _encode_piece(piece: Tuple[int,int,int,int]) -> int:
        return MCTSNode.pieces.index(piece) + 1

    @staticmethod
    def _decode_piece(val: int) -> Tuple[int,int,int,int]:
        return MCTSNode.pieces[val-1]


class P1:
    """
    MCTS-based player with parallel search and risk avoidance.
    """
    MAX_TURN_TIME = 5
    ITERATION_CAP = 500

    def __init__(
        self,
        board: List[List[int]],
        available_pieces: List[Tuple[int,int,int,int]]
    ):
        self.exploration_base = 1.4
        self.heuristic_weight_base = 0.6
        self._adjust_parameters(board)
        self.root = MCTSNode(board, available_pieces, 'select')

    def _adjust_parameters(self, board: List[List[int]]):
        empty = sum(cell == 0 for row in board for cell in row)
        progress = 1.0 - (empty / 16.0)
        self.exploration = self.exploration_base * (1.0 - 0.3 * progress)
        self.heuristic_weight = self.heuristic_weight_base * (1.0 + 0.5 * progress)

    def _search(self):
        """
        Parallel MCTS search using small batches and FIRST_COMPLETED wait.
        Records actual iterations performed.
        """
        iterations = 0
        end_time = time.time() + P1.MAX_TURN_TIME * 0.9
        batch_size = 10
        with ThreadPoolExecutor(max_workers=4) as executor:
            while time.time() < end_time and iterations < P1.ITERATION_CAP:
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                cb = min(batch_size, P1.ITERATION_CAP - iterations)
                futures = [executor.submit(self._iterate) for _ in range(cb)]
                done, _ = wait(
                    futures,
                    timeout=min(0.05, remaining),
                    return_when=FIRST_COMPLETED
                )
                for f in done:
                    try:
                        f.result()
                        iterations += 1
                    except:
                        pass
        self.last_search_iterations = iterations

    def _iterate(self):
        node = self.root
        # selection
        while not node.untried_actions and node.children:
            node = node.best_child(self.exploration, self.heuristic_weight)
        # expansion
        if node.untried_actions:
            node = node.expand()
        # simulation & backpropagation
        result = node.simulate()
        node.backpropagate(result)

    def select_piece(self) -> Tuple[int,int,int,int]:
        """
        Chooses a piece to hand over to opponent.
        """
        self._search()
        if not self.root.children:
            choice = random.choice(self.root.available_pieces)
        else:
            best = max(self.root.children, key=lambda c: c.visits)
            self.root = best
            choice = best.selected_piece
        print(f"SELECT_PIECE -> {choice}, iters={self.last_search_iterations}")
        return choice

    def place_piece(
        self,
        piece: Tuple[int,int,int,int]
    ) -> Tuple[int,int]:
        """
        Places the given piece on board, preferring winning or safe moves.
        """
        # immediate win
        for r, c in product(range(4), range(4)):
            if self.root.board[r][c] == 0:
                board_copy = [row.copy() for row in self.root.board]
                board_copy[r][c] = MCTSNode._encode_piece(piece)
                if MCTSNode._check_win(board_copy):
                    print(f"PLACE_PIECE -> ({r},{c}), iters={self.last_search_iterations}")
                    return (r, c)
        # identify dangerous spots
        danger = []
        for r, c in product(range(4), range(4)):
            if self.root.board[r][c] == 0:
                board_copy = [row.copy() for row in self.root.board]
                board_copy[r][c] = MCTSNode._encode_piece(piece)
                key2 = tuple(tuple(row) for row in board_copy)
                if any(
                    p != piece and MCTSNode._opponent_can_win_cached(key2, p)
                    for p in self.root.available_pieces
                ):
                    danger.append((r, c))
        # reuse or reset
        for ch in self.root.children:
            if ch.selected_piece == piece:
                ch.parent = None
                self.root = ch
                break
        else:
            self.root = MCTSNode(
                self.root.board,
                self.root.available_pieces,
                'place',
                selected_piece=piece
            )
        self._search()
        # choose best safe move
        if not self.root.children:
            empties = [
                (r, c)
                for r, c in product(range(4), range(4))
                if self.root.board[r][c] == 0
            ]
            safe = [pos for pos in empties if pos not in danger]
            choice = safe[0] if safe else empties[0]
        else:
            best = max(self.root.children, key=lambda c: c.visits)
            candidate = next(
                (
                    pos for pos in product(range(4), range(4))
                    if self.root.board[pos[0]][pos[1]] == 0 and best.board[pos[0]][pos[1]] != 0
                ),
                (0, 0)
            )
            if candidate in danger and len(self.root.children) > 1:
                second = sorted(self.root.children, key=lambda c: c.visits)[-2]
                alt = next(
                    (
                        pos for pos in product(range(4), range(4))
                        if self.root.board[pos[0]][pos[1]] == 0
                        and second.board[pos[0]][pos[1]] != 0
                        and pos not in danger
                    ),
                    candidate
                )
                choice = alt
            else:
                choice = candidate
        print(f"PLACE_PIECE -> {choice}, iters={self.last_search_iterations}")
        return choice

    def record_game(self, won: bool):
        """Placeholder for recording game outcome."""
        pass