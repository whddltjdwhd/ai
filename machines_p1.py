import random
import math
import time
import pickle
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
    
    # 시뮬레이션 깊이 설정 (동적으로 조정 가능)
    SIMULATION_DEPTH = 200

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
        
        # 중앙 위치 가중치
        for r, c in [(1,1),(1,2),(2,1),(2,2)]:
            if board[r][c] == 0:
                score += 0.1
        
        # 각 속성별(I/E, N/S, T/F, P/J) 분석 추가
        for attr_idx in range(4):
            # 각 라인에서 속성 일치 체크
            for line in MCTSNode._all_lines(board):
                attr_counts = [0, 0]  # [0속성 개수, 1속성 개수]
                empty_count = 0
                for val in line:
                    if val == 0:
                        empty_count += 1
                    else:
                        piece = MCTSNode._decode_piece(val)
                        attr_counts[piece[attr_idx]] += 1
                
                # 3개 같은 속성이 있고 1개 빈칸 - 잠재적 승리 가능성
                if empty_count == 1 and (attr_counts[0] == 3 or attr_counts[1] == 3):
                    score += 0.3
                # 2개 같은 속성이 있고 2개 빈칸 - 잠재적 가능성
                elif empty_count == 2 and (attr_counts[0] == 2 or attr_counts[1] == 2):
                    score += 0.1
        
        # 2x2 블록 분석 추가
        for r in range(3):
            for c in range(3):
                for attr_idx in range(4):
                    block = [
                        board[r][c], board[r][c+1],
                        board[r+1][c], board[r+1][c+1]
                    ]
                    attr_counts = [0, 0]
                    empty_count = 0
                    for val in block:
                        if val == 0:
                            empty_count += 1
                        else:
                            piece = MCTSNode._decode_piece(val)
                            attr_counts[piece[attr_idx]] += 1
                    
                    # 3개 같은 속성이 있고 1개 빈칸
                    if empty_count == 1 and (attr_counts[0] == 3 or attr_counts[1] == 3):
                        score += 0.25
        
        # 기존 점수 반영
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
        piece: Tuple[int, int, int, int]
    ) -> bool:
        """
        Fast check if opponent can win by placing `piece` anywhere.
        """
        board = [list(row) for row in board_key]
        encoded_piece = MCTSNode._encode_piece(piece)
        
        # 모든 빈 위치에 대해 체크
        for r, c in product(range(4), range(4)):
            if board[r][c] == 0:
                # 피스를 임시로 배치
                board[r][c] = encoded_piece
                
                # 승리 체크
                if MCTSNode._check_win(board):
                    # 복원
                    board[r][c] = 0
                    return True
                    
                # 양방 3목 체크 (더 위험한 상황)
                fork_count = 0
                for attr_idx in range(4):
                    # 가로줄 체크
                    row_match = all(board[r][i] != 0 and MCTSNode._decode_piece(board[r][i])[attr_idx] == 
                                   piece[attr_idx] for i in range(4))
                    if row_match:
                        fork_count += 1
                    
                    # 세로줄 체크
                    col_match = all(board[i][c] != 0 and MCTSNode._decode_piece(board[i][c])[attr_idx] == 
                                   piece[attr_idx] for i in range(4))
                    if col_match:
                        fork_count += 1
                    
                    # 대각선 체크 (해당되는 경우만)
                    if r == c:
                        diag_match = all(board[i][i] != 0 and MCTSNode._decode_piece(board[i][i])[attr_idx] == 
                                        piece[attr_idx] for i in range(4))
                        if diag_match:
                            fork_count += 1
                            
                    if r + c == 3:
                        anti_diag_match = all(board[i][3-i] != 0 and MCTSNode._decode_piece(board[i][3-i])[attr_idx] == 
                                             piece[attr_idx] for i in range(4))
                        if anti_diag_match:
                            fork_count += 1
                
                # 여러 승리 경로가 있으면 양방 3목으로 판단
                if fork_count >= 2:
                    board[r][c] = 0
                    return True
                
                # 복원
                board[r][c] = 0
        
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
        board_copy = [row.copy() for row in self.board]
        pieces = self.available_pieces.copy()
        phase = self.player_phase
        current = self.selected_piece
        
        # 즉시 승리 체크 - 시뮬레이션 시작 전
        if current and phase == 'place':
            for r, c in product(range(4), range(4)):
                if board_copy[r][c] == 0:
                    board_copy[r][c] = MCTSNode._encode_piece(current)
                    if MCTSNode._check_win(board_copy):
                        return 1.0  # 즉시 승리
                    board_copy[r][c] = 0  # 복원
        
        # 시뮬레이션 실행
        for depth in range(MCTSNode.SIMULATION_DEPTH):
            # 게임 종료 체크
            if MCTSNode._check_win(board_copy):
                return 1.0 if depth % 2 == 0 else 0.0
                
            if not pieces:
                all_filled = all(cell != 0 for row in board_copy for cell in row)
                if all_filled:
                    return 0.5  # 무승부
            
            if phase == 'select':
                # 안전한 피스 선택 (양방 3목 방지)
                board_key = tuple(tuple(r) for r in board_copy)
                safe = [p for p in pieces if not MCTSNode._opponent_can_win_cached(board_key, p)]
                
                # 안전한 피스가 없으면 기존 피스 중에서 선택
                if not safe and pieces:
                    choice = random.choice(pieces)
                elif safe:
                    choice = random.choice(safe)
                else:
                    return 0.5  # 더 이상 선택할 피스 없음
                    
                pieces.remove(choice)
                current = choice
                phase = 'place'
            else:  # phase == 'place'
                # 양방 3목 기회 찾기
                empties = [(r, c) for r, c in product(range(4), range(4)) if board_copy[r][c] == 0]
                if not empties:
                    return 0.5  # 더 이상 둘 곳 없음
                    
                # 승리 가능한 위치 찾기
                winning_moves = []
                fork_moves = []
                
                for r, c in empties:
                    board_copy[r][c] = MCTSNode._encode_piece(current)
                    if MCTSNode._check_win(board_copy):
                        winning_moves.append((r, c))
                    else:
                        # 양방 3목 체크
                        fork_count = MCTSNode._check_fork_opportunities(board_copy, r, c, current)
                        if fork_count >= 2:
                            fork_moves.append((r, c))
                    board_copy[r][c] = 0  # 복원
                
                if winning_moves:
                    r, c = random.choice(winning_moves)
                elif fork_moves:
                    r, c = random.choice(fork_moves)
                else:
                    r, c = random.choice(empties)
                    
                board_copy[r][c] = MCTSNode._encode_piece(current)
                phase = 'select'
        
        # 기본 평가 (게임 상태 평가)
        return 0.5 + 0.1 * MCTSNode._evaluate_state(
            tuple(tuple(r) for r in board_copy),
            tuple(pieces)
        )

    @staticmethod
    def _check_fork_opportunities(board: List[List[int]], row: int, col: int, piece: Tuple[int, int, int, int]) -> int:
        """위치에 피스를 놓았을 때 생기는 양방 3목 기회 계산"""
        potential_wins = 0
        for attr_idx in range(4):
            for line_type in ['row', 'col', 'diag']:
                if line_type == 'row':
                    line = [board[row][c] for c in range(4)]
                elif line_type == 'col':
                    line = [board[r][col] for r in range(4)]
                elif line_type == 'diag' and row == col:  # 주대각선
                    line = [board[i][i] for i in range(4)]
                elif line_type == 'diag' and row + col == 3:  # 부대각선
                    line = [board[i][3-i] for i in range(4)]
                else:
                    continue
                    
                # 해당 라인이 특정 속성에서 3개 일치하는지 확인
                if 0 not in line:  # 라인에 빈칸이 없어야 함
                    pieces = []
                    for val in line:
                        if val != 0:
                            pieces.append(MCTSNode._decode_piece(val))
                    
                    if all(p[attr_idx] == pieces[0][attr_idx] for p in pieces):
                        potential_wins += 1
        
        # 2x2 블록 체크
        if 1 <= row <= 2 and 1 <= col <= 2:
            for r_offset in [-1, 0]:
                for c_offset in [-1, 0]:
                    if 0 <= row+r_offset <= 2 and 0 <= col+c_offset <= 2:
                        block = [
                            board[row+r_offset][col+c_offset], 
                            board[row+r_offset][col+c_offset+1],
                            board[row+r_offset+1][col+c_offset], 
                            board[row+r_offset+1][col+c_offset+1]
                        ]
                        if 0 not in block:
                            for attr_idx in range(4):
                                pieces = []
                                for val in block:
                                    if val != 0:
                                        pieces.append(MCTSNode._decode_piece(val))
                                
                                if all(p[attr_idx] == pieces[0][attr_idx] for p in pieces):
                                    potential_wins += 1
        
        return potential_wins

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
    ITERATION_CAP = 2000
    
    # 게임 기록 관리를 위한 변수 추가
    move_history = []
    win_patterns = {}

    def __init__(
        self,
        board: List[List[int]],
        available_pieces: List[Tuple[int,int,int,int]]
    ):
        # Determine first/second player
        is_first = len(available_pieces) % 2 == 0
        self.exploration_base = 1.4 if is_first else 1.6
        self.heuristic_weight_base = 0.35 if is_first else 0.25
        
        # 학습된 패턴 로드
        self._load_patterns()
        
        # Adjust for game stage
        self._adjust_params(board)
        self.root = MCTSNode(board, available_pieces, 'select')
        self.last_search_iterations = 0

    def _adjust_params(self, board: List[List[int]]):
        empty = sum(cell == 0 for row in board for cell in row)
        
        # 게임 초반 (16-12 빈칸)
        if empty >= 12:
            self.exploration = self.exploration_base * 1.2
            self.heuristic_weight = self.heuristic_weight_base * 0.8
            MCTSNode.SIMULATION_DEPTH = 100  # 초반에는 다양하게 탐색
            
        # 게임 중반 (11-6 빈칸)
        elif empty >= 6:
            self.exploration = self.exploration_base
            self.heuristic_weight = self.heuristic_weight_base * 1.2
            MCTSNode.SIMULATION_DEPTH = 200  # 중반에는 균형있게 탐색
            
        # 게임 후반 (5-0 빈칸)
        else:
            self.exploration = self.exploration_base * 0.7
            self.heuristic_weight = self.heuristic_weight_base * 1.5
            MCTSNode.SIMULATION_DEPTH = 300  # 후반에는 깊게 탐색하여 승리 기회 포착

    def _danger_level(self, pos: Tuple[int,int], piece: Tuple[int,int,int,int]) -> int:
        # Evaluate risk if piece placed at pos
        board_copy = [row.copy() for row in self.root.board]
        board_copy[pos[0]][pos[1]] = MCTSNode._encode_piece(piece)
        key = tuple(tuple(r) for r in board_copy)
        
        # 양방 3목 체크
        fork_count = MCTSNode._check_fork_opportunities(board_copy, pos[0], pos[1], piece)
        if fork_count >= 2:
            return 5  # 양방 3목은 가장 위험
        
        # 피스 위험도 체크
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
        end_time = time.time() + self.MAX_TURN_TIME * 0.95
        iters = 0
        future_results = []
        
        # 초기에 더 많은 스레드로 탐색 시작
        with ThreadPoolExecutor(max_workers=8) as executor:
            while time.time() < end_time and iters < self.ITERATION_CAP:
                # 배치 단위로 작업 제출 (CPU 코어별 최적화)
                batch_size = 32  # 한 번에 제출할 작업 개수
                new_futures = [executor.submit(self._iterate) for _ in range(batch_size)]
                future_results.extend(new_futures)
                
                # 완료된 작업 확인 및 카운트
                done, _ = wait(future_results, timeout=0.01)
                for future in done:
                    future_results.remove(future)
                    try:
                        future.result()
                        iters += 1
                    except:
                        pass
                
                # 일정 간격으로 가장 유망한 경로에 집중
                if iters % 200 == 0 and self.root.children:
                    # 가장 많이 방문된 자식 노드를 새로운 루트로 설정
                    best_child = max(self.root.children, key=lambda c: c.visits)
                    if best_child.visits > self.root.visits * 0.5:  # 충분히 유망한 경우만
                        self.root = best_child
                
        self.last_search_iterations = iters

    def _analyze_opponent_trends(self, board, available_pieces):
        """P2의 전략 경향을 분석하여 대응"""
        # P2가 MBTI 기반 접근법을 사용한다고 가정하고 대응
        
        # 속성별 분포 분석
        attr_distribution = []
        for i in range(4):  # 각 속성별
            zeros = sum(1 for r in range(4) for c in range(4) 
                       if board[r][c] != 0 and MCTSNode._decode_piece(board[r][c])[i] == 0)
            ones = sum(1 for r in range(4) for c in range(4) 
                      if board[r][c] != 0 and MCTSNode._decode_piece(board[r][c])[i] == 1)
            
            # P2가 선호하는 속성 파악
            if zeros > ones + 1:
                # 0속성(I/N/T/P) 선호 경향
                attr_distribution.append((0, 0.7))
            elif ones > zeros + 1:
                # 1속성(E/S/F/J) 선호 경향
                attr_distribution.append((1, 0.7))
            else:
                # 뚜렷한 선호 없음
                attr_distribution.append((None, 0.5))
        
        return attr_distribution

    def _adjust_children_by_trends(self, attr_distribution):
        """상대방 경향에 따른 자식 노드 조정"""
        if not self.root.children:
            return
            
        for child in self.root.children:
            piece = child.selected_piece
            if piece:
                # P2가 선호하는 속성의 피스에 패널티
                for i, (preferred, weight) in enumerate(attr_distribution):
                    if preferred is not None and piece[i] == preferred:
                        child.heuristic -= 0.1 * weight

    def _check_fork_opportunities(self, board: List[List[int]], pos: Tuple[int, int], piece: Tuple[int, int, int, int]) -> bool:
        """위치에 피스를 놓았을 때 양방 3목 기회가 있는지 확인"""
        board_copy = [row.copy() for row in board]
        board_copy[pos[0]][pos[1]] = MCTSNode._encode_piece(piece)
        
        fork_count = MCTSNode._check_fork_opportunities(board_copy, pos[0], pos[1], piece)
        return fork_count >= 2

    def select_piece(self) -> Tuple[int,int,int,int]:
        empty = sum(cell == 0 for row in self.root.board for cell in row)
        
        # 상대 전략 분석
        attr_distribution = self._analyze_opponent_trends(self.root.board, self.root.available_pieces)
        
        # Early game: danger-based selection with attribute analysis
        if empty >= 12:
            empties = [(r, c) for r, c in product(range(4), range(4)) if self.root.board[r][c] == 0]
            # 피스별 위험도 평가
            danger_scores = {}
            for p in self.root.available_pieces:
                # 기본 위험도
                danger = max(self._danger_level(e, p) for e in empties)
                
                # 학습된 패턴 기반 가중치 조정
                pattern_key = self._encode_pattern(p)
                if pattern_key in self.win_patterns:
                    danger -= min(self.win_patterns[pattern_key] * 0.1, 1)  # 승패 기록 반영
                
                # 상대 선호 속성과 일치하는 경우 위험도 증가
                for i, (preferred, weight) in enumerate(attr_distribution):
                    if preferred is not None and p[i] == preferred:
                        danger += 1 * weight
                        
                danger_scores[p] = danger
                
            return min(self.root.available_pieces, key=lambda p: danger_scores[p])
            
        # Mid/late game: MCTS with enhanced analysis
        self._search()
        
        if not self.root.children:
            safest = min(self.root.available_pieces, 
                         key=lambda p: sum(1 for e in product(range(4), range(4)) 
                                         if self.root.board[e[0]][e[1]] == 0 
                                         and MCTSNode._opponent_can_win_cached(
                                             tuple(tuple(r) for r in self.root.board), p)))
            return safest
            
        # 상대 경향에 따른 자식 노드 조정
        self._adjust_children_by_trends(attr_distribution)
        
        best = max(self.root.children, key=lambda c: c.visits)
        self.root = best
        choice = best.selected_piece
        
        # 선택을 기록
        self.move_history.append(choice)
        
        print(f"SELECT_PIECE -> {choice}, iters={self.last_search_iterations}")
        return choice

    def place_piece(self, piece: Tuple[int,int,int,int]) -> Tuple[int,int]:
        # Immediate win if available
        for r, c in product(range(4), range(4)):
            if self.root.board[r][c] == 0:
                temp = [row.copy() for row in self.root.board]
                temp[r][c] = MCTSNode._encode_piece(piece)
                if MCTSNode._check_win(temp):
                    # 승리 위치 기록
                    self.move_history.append((r, c))
                    print(f"PLACE_PIECE -> ({r},{c}), iters={self.last_search_iterations}")
                    return (r, c)

        # 양방 3목 기회 찾기
        fork_opportunities = []
        for r, c in product(range(4), range(4)):
            if self.root.board[r][c] == 0:
                if self._check_fork_opportunities(self.root.board, (r, c), piece):
                    fork_opportunities.append((r, c))
                    
        if fork_opportunities:
            choice = random.choice(fork_opportunities)
            self.move_history.append(choice)
            print(f"PLACE_PIECE [FORK] -> {choice}")
            return choice
                
        # Identify dangerous spots
        danger_spots = [(r, c) for r, c in product(range(4), range(4)) 
                       if self.root.board[r][c]==0 and self._danger_level((r,c), piece)>0]
                       
        # Search with enhanced strategy
        self._search()
        
        # No children: pick safe or first empty
        if not self.root.children:
            empties = [(r,c) for r,c in product(range(4), range(4)) if self.root.board[r][c]==0]
            safe = [e for e in empties if e not in danger_spots]
            
            # 중앙에 가까운 안전한 위치 선호
            if safe:
                safe.sort(key=lambda pos: abs(pos[0]-1.5) + abs(pos[1]-1.5))
                choice = safe[0]
            else:
                empties.sort(key=lambda pos: abs(pos[0]-1.5) + abs(pos[1]-1.5))
                choice = empties[0]
                
            self.move_history.append(choice)
            print(f"PLACE_PIECE -> {choice}, iters={self.last_search_iterations}")
            return choice
            
        # Use child visitation to pick move not in danger
        best = max(self.root.children, key=lambda c: c.visits)
        
        # 최적의 위치 찾기
        for r, c in product(range(4), range(4)):
            if best.board[r][c]!=self.root.board[r][c] and self.root.board[r][c]==0:
                if (r,c) not in danger_spots:
                    self.move_history.append((r, c))
                    print(f"PLACE_PIECE -> ({r},{c}), iters={self.last_search_iterations}")
                    return (r, c)
                    
        # Fallback - 최대한 중앙에 가까운 위치 선택
        empties = [(r,c) for r,c in product(range(4), range(4)) if self.root.board[r][c]==0]
        if empties:
            choice = min(empties, key=lambda pos: abs(pos[0]-1.5) + abs(pos[1]-1.5))
            self.move_history.append(choice)
            print(f"PLACE_PIECE -> {choice}, iters={self.last_search_iterations}")
            return choice
            
        # 더 이상 둘 곳 없음
        fallback = next((e for e in product(range(4), range(4)) if self.root.board[e[0]][e[1]]==0), (0, 0))
        self.move_history.append(fallback)
        print(f"PLACE_PIECE -> {fallback}, iters={self.last_search_iterations}")
        return fallback

    def _encode_pattern(self, move):
        """게임 패턴을 고유 키로 인코딩"""
        if isinstance(move, tuple):
            if len(move) == 4:  # 피스 선택
                return f"select:{move}"
            else:  # 피스 배치
                return f"place:{move[0]},{move[1]}"
        return f"unknown:{move}"
            
    def _save_patterns(self):
        """승률 높은 패턴 저장"""
        try:
            with open('winning_patterns.dat', 'wb') as f:
                pickle.dump(self.win_patterns, f)
        except:
            pass
            
    def _load_patterns(self):
        """저장된 패턴 불러오기"""
        try:
            with open('winning_patterns.dat', 'rb') as f:
                self.win_patterns = pickle.load(f)
        except:
            self.win_patterns = {}

    def record_game(self, won: bool):
        """게임 종료 후 기록 저장"""
        if won:
            # 승리한 게임의 패턴을 기록
            for move in self.move_history:
                pattern_key = self._encode_pattern(move)
                if pattern_key in self.win_patterns:
                    self.win_patterns[pattern_key] += 1
                else:
                    self.win_patterns[pattern_key] = 1
        
        # 게임 히스토리 초기화
        self.move_history = []
        
        # 승률이 높은 패턴을 파일에 저장
        if random.random() < 0.1:  # 10% 확률로 저장 (성능 최적화)
            self._save_patterns()