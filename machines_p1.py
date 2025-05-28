import random
import math
import time
import pickle
import os
from itertools import product
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import List, Tuple, Optional, Any, Dict

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
    
    # 위험도 캐싱
    danger_cache = {}

    @staticmethod
    @lru_cache(maxsize=16384)
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
        
        # 1. 각 속성별(I/E, N/S, T/F, P/J) 분석
        for attr_idx in range(4):
            # 각 라인에서 속성 일치 체크
            for line in MCTSNode._all_lines(board):
                attr_counts = [0, 0]  # [0속성 개수, 1속성 개수]
                empty_count = 0
                empty_pos = None
                
                for pos, val in enumerate(line):
                    if val == 0:
                        empty_count += 1
                        empty_pos = pos
                    else:
                        piece = MCTSNode._decode_piece(val)
                        attr_counts[piece[attr_idx]] += 1
                
                # 3개 같은 속성이 있고 1개 빈칸 - 매우 높은 점수
                if empty_count == 1:
                    if attr_counts[0] == 3:
                        score += 0.5  # 매우 중요한 패턴
                    elif attr_counts[1] == 3:
                        score += 0.5
                # 2개 같은 속성이 있고 2개 빈칸 - 잠재적 패턴
                elif empty_count == 2:
                    if attr_counts[0] == 2:
                        score += 0.2
                    elif attr_counts[1] == 2:
                        score += 0.2
        
        # 2. 2x2 블록 승리 패턴 분석
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
                    
                    # 3개 같은 속성이 있고 1개 빈칸 - 높은 점수
                    if empty_count == 1:
                        if attr_counts[0] == 3 or attr_counts[1] == 3:
                            score += 0.6  # Changed from 0.4
                    # 2개 같은 속성, 2개 빈칸 - 중간 점수
                    elif empty_count == 2:
                        if attr_counts[0] == 2 or attr_counts[1] == 2:
                            score += 0.25  # Changed from 0.15
        
        # 3. 중앙 위치 가중치 (기본적인 전략)
        center_control = 0
        for r, c in [(1,1), (1,2), (2,1), (2,2)]:
            if board[r][c] != 0:
                center_control += 1
        score += center_control * 0.05  # 중앙 통제는 소폭 점수 부여
        
        return score

    @staticmethod
    @lru_cache(maxsize=32768)
    def _opponent_can_win_cached(
        board_key: Tuple[Tuple[int, ...], ...],
        piece: Tuple[int, int, int, int]
    ) -> int:
        """
        Fast check if opponent can win by placing `piece` anywhere.
        Returns:
            0: 안전함
            1: 승리 가능
            2: 양방 3목 가능 (더 위험)
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
                    board[r][c] = 0
                    return 1
                    
                # 양방 3목 체크
                fork_count = MCTSNode._check_fork_opportunities(board, r, c, piece)
                if fork_count >= 2:
                    board[r][c] = 0
                    return 2
                
                # 복원
                board[r][c] = 0
        
        return 0

    def __init__(
        self,
        board: List[List[int]],
        available_pieces: List[Tuple[int,int,int,int]],
        player_phase: str,
        selected_piece: Optional[Tuple[int,int,int,int]] = None,
        parent: Optional['MCTSNode'] = None,
        action_taken: Optional[Any] = None # Action that led to this node from parent
    ):
        self.board = [row.copy() for row in board]
        self.available_pieces = available_pieces.copy()
        self.player_phase = player_phase  # 'select' or 'place'
        self.selected_piece = selected_piece # If phase is 'place', this is the piece to place. If phase is 'select', this is None initially.
        self.parent = parent
        self.action_taken = action_taken # The action (piece or position) that led from parent to this node
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
        next_board, next_avail, next_phase, next_selected_piece_for_child = self._apply_action(action)
        
        child = MCTSNode(
            next_board, 
            next_avail, 
            next_phase, 
            selected_piece=next_selected_piece_for_child, 
            parent=self,
            action_taken=action # Store the action that led to this child
        )
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
                
                # 위험도에 따라 피스 정렬 (안전한 것 우선)
                piece_dangers = []
                for p in pieces:
                    danger = MCTSNode._opponent_can_win_cached(board_key, p)
                    piece_dangers.append((p, danger))
                
                # 안전한 피스 필터링
                safe_pieces = [p for p, danger in piece_dangers if danger == 0]
                
                if safe_pieces:
                    # 안전한 피스 중 속성 패턴이 좋은 것 선택
                    choice = random.choice(safe_pieces)
                elif pieces:
                    # 모든 피스가 위험하면, 덜 위험한 것 선택
                    less_dangerous = [p for p, danger in piece_dangers if danger == 1]
                    if less_dangerous:
                        choice = random.choice(less_dangerous)
                    else:
                        choice = random.choice(pieces)
                else:
                    return 0.5  # 더 이상 선택할 피스 없음
                    
                pieces.remove(choice)
                current = choice
                phase = 'place'
            else:  # phase == 'place'
                empties = [(r, c) for r, c in product(range(4), range(4)) if board_copy[r][c] == 0]
                if not empties:
                    return 0.5  # 더 이상 둘 곳 없음
                
                # 1. 즉시 승리 가능한 위치 찾기
                winning_moves = []
                fork_moves = []
                
                for r, c in empties:
                    board_copy[r][c] = MCTSNode._encode_piece(current)
                    
                    # 승리 체크
                    if MCTSNode._check_win(board_copy):
                        winning_moves.append((r, c))
                    else:
                        # 양방 3목 체크
                        fork_count = MCTSNode._check_fork_opportunities(board_copy, r, c, current)
                        if fork_count >= 2:
                            fork_moves.append((r, c))
                    
                    board_copy[r][c] = 0  # 복원
                
                # 2. 최적의 수 선택
                if winning_moves:  # 승리 위치
                    r, c = random.choice(winning_moves)
                elif fork_moves:   # 양방 3목 위치
                    r, c = random.choice(fork_moves)
                else:  # 기본 선택 - 중앙 선호
                    central = []
                    corners = []
                    others = []
                    
                    for pos in empties:
                        r, c = pos
                        if (r, c) in [(1,1), (1,2), (2,1), (2,2)]:
                            central.append(pos)
                        elif (r, c) in [(0,0), (0,3), (3,0), (3,3)]:
                            corners.append(pos)
                        else:
                            others.append(pos)
                    
                    if central:
                        r, c = random.choice(central)
                    elif corners:
                        r, c = random.choice(corners)
                    else:
                        r, c = random.choice(others)
                
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
        encoded_piece = MCTSNode._encode_piece(piece)
        
        # 1. 속성별로 승리 라인 체크
        for attr_idx in range(4):
            # 행 체크
            row_pieces = [board[row][c] for c in range(4)]
            if row_pieces.count(0) == 0:  # 모두 채워진 경우
                row_attr_match = True
                for val in row_pieces:
                    if val != encoded_piece:
                        decoded = MCTSNode._decode_piece(val)
                        if decoded[attr_idx] != piece[attr_idx]:
                            row_attr_match = False
                            break
                if row_attr_match:
                    potential_wins += 1
            
            # 열 체크
            col_pieces = [board[r][col] for r in range(4)]
            if col_pieces.count(0) == 0:  # 모두 채워진 경우
                col_attr_match = True
                for val in col_pieces:
                    if val != encoded_piece:
                        decoded = MCTSNode._decode_piece(val)
                        if decoded[attr_idx] != piece[attr_idx]:
                            col_attr_match = False
                            break
                if col_attr_match:
                    potential_wins += 1
            
            # 대각선 체크 (해당되는 경우만)
            if row == col:  # 주 대각선
                diag_pieces = [board[i][i] for i in range(4)]
                if diag_pieces.count(0) == 0:
                    diag_attr_match = True
                    for val in diag_pieces:
                        if val != encoded_piece:
                            decoded = MCTSNode._decode_piece(val)
                            if decoded[attr_idx] != piece[attr_idx]:
                                diag_attr_match = False
                                break
                    if diag_attr_match:
                        potential_wins += 1
            
            if row + col == 3:  # 반대 대각선
                anti_diag_pieces = [board[i][3-i] for i in range(4)]
                if anti_diag_pieces.count(0) == 0:
                    anti_diag_attr_match = True
                    for val in anti_diag_pieces:
                        if val != encoded_piece:
                            decoded = MCTSNode._decode_piece(val)
                            if decoded[attr_idx] != piece[attr_idx]:
                                anti_diag_attr_match = False
                                break
                    if anti_diag_attr_match:
                        potential_wins += 1
        
        # 2. 2x2 블록 승리 체크
        if 0 <= row <= 2 and 0 <= col <= 2:  # 왼쪽 상단 블록
            block = [board[row][col], board[row][col+1], board[row+1][col], board[row+1][col+1]]
            if 0 not in block:
                for attr_idx in range(4):
                    attr_match = True
                    for val in block:
                        if val != encoded_piece:
                            decoded = MCTSNode._decode_piece(val)
                            if decoded[attr_idx] != piece[attr_idx]:
                                attr_match = False
                                break
                    if attr_match:
                        potential_wins += 1
        
        if 0 <= row-1 and row <= 3 and 0 <= col <= 2:  # 오른쪽 상단 블록
            block = [board[row-1][col], board[row-1][col+1], board[row][col], board[row][col+1]]
            if 0 not in block:
                for attr_idx in range(4):
                    attr_match = True
                    for val in block:
                        if val != encoded_piece:
                            decoded = MCTSNode._decode_piece(val)
                            if decoded[attr_idx] != piece[attr_idx]:
                                attr_match = False
                                break
                    if attr_match:
                        potential_wins += 1
        
        if 0 <= row <= 2 and 0 <= col-1 and col <= 3:  # 왼쪽 하단 블록
            block = [board[row][col-1], board[row][col], board[row+1][col-1], board[row+1][col]]
            if 0 not in block:
                for attr_idx in range(4):
                    attr_match = True
                    for val in block:
                        if val != encoded_piece:
                            decoded = MCTSNode._decode_piece(val)
                            if decoded[attr_idx] != piece[attr_idx]:
                                attr_match = False
                                break
                    if attr_match:
                        potential_wins += 1
        
        if 0 <= row-1 and row <= 3 and 0 <= col-1 and col <= 3:  # 오른쪽 하단 블록
            block = [board[row-1][col-1], board[row-1][col], board[row][col-1], board[row][col]]
            if 0 not in block:
                for attr_idx in range(4):
                    attr_match = True
                    for val in block:
                        if val != encoded_piece:
                            decoded = MCTSNode._decode_piece(val)
                            if decoded[attr_idx] != piece[attr_idx]:
                                attr_match = False
                                break
                    if attr_match:
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
        """모든 승리 가능 라인 반환 (가로, 세로, 대각선)"""
        lines: List[List[int]] = []
        # 가로줄
        for i in range(4):
            lines.append([board[i][j] for j in range(4)])
        # 세로줄
        for i in range(4):
            lines.append([board[j][i] for j in range(4)])
        # 주대각선
        lines.append([board[i][i] for i in range(4)])
        # 반대각선
        lines.append([board[i][3-i] for i in range(4)])
        return lines

    @staticmethod
    def _check_win(board: List[List[int]]) -> bool:
        """승리 조건 체크 (4개 일치 또는 2x2 블록 일치)"""
        # 모든 라인 체크
        for line in MCTSNode._all_lines(board):
            if 0 not in line:  # 빈칸 없이 채워진 경우만
                traits = [MCTSNode._decode_piece(v) for v in line]
                # 4개 속성 중 하나라도 모두 일치하면 승리
                if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)):
                    return True
        
        # 2x2 블록 체크
        for r in range(3):
            for c in range(3):
                block = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in block:  # 빈칸 없이 채워진 경우만
                    traits = [MCTSNode._decode_piece(v) for v in block]
                    # 4개 속성 중 하나라도 모두 일치하면 승리
                    if any(all(t[i] == traits[0][i] for t in traits) for i in range(4)):
                        return True
        return False

    @staticmethod
    def _encode_piece(piece: Tuple[int,int,int,int]) -> int:
        """피스 인코딩 (튜플 → 정수)"""
        return MCTSNode.pieces.index(piece) + 1

    @staticmethod
    def _decode_piece(val: int) -> Tuple[int,int,int,int]:
        """피스 디코딩 (정수 → 튜플)"""
        return MCTSNode.pieces[val-1]

    @staticmethod
    def _get_mbti_name(piece: Tuple[int,int,int,int]) -> str:
        """피스를 MBTI 형식으로 표현"""
        return f"{'I' if piece[0] == 0 else 'E'}{'N' if piece[1] == 0 else 'S'}{'T' if piece[2] == 0 else 'F'}{'P' if piece[3] == 0 else 'J'}"

class P1:
    """
    MCTS-based player with enhanced strategy against P2's approach.
    """
    
    # 게임 기록 관리를 위한 변수
    _instance_move_history = []  # 인스턴스별 기록
    _global_win_patterns = {}    # 전역 승리 패턴
    
    # 디버깅 모드를 클래스 변수로 설정
    debug = True
    
    def _turn_time(self):
        empty = sum(cell==0 for row in self.board for cell in row)
        if empty >= 12:
            return 5
        elif empty >= 6:
            return 15
        else:
            return 25

        
    def __init__(
        self,
        board: List[List[int]],
        available_pieces: List[Tuple[int,int,int,int]],
        avg_iters_per_sec: float = 500
    ):
        # 게임 상태 초기화
        self.board = [row.copy() for row in board]
        self.available_pieces = available_pieces.copy()
        self.avg_iters_per_sec = avg_iters_per_sec
        
        self.MAX_TURN_TIME = self._turn_time()
        self.ITERATION_CAP = int(self.avg_iters_per_sec * self.MAX_TURN_TIME * 0.95)
        
        # 선공/후공 판단 및 기본 파라미터 설정
        self.is_first = len(available_pieces) % 2 == 0
        self.exploration_base = 1.6 if self.is_first else 1.8
        self.heuristic_weight_base = 0.4 if self.is_first else 0.3
        
        # 인스턴스 변수 초기화
        self.move_history = []
        
        # 게임 단계에 따른 파라미터 조정
        self._adjust_params(board)
        
        # MCTS 루트 노드 초기화
        self.root = MCTSNode(board, available_pieces, 'select')
        
        # 탐색 통계
        self.last_search_iterations = 0
        self.last_search_time = 0
        
        # 디버깅 모드 - 인스턴스마다 개별 설정 가능하도록 유지
        # self.debug = True  # 이 줄을 주석 처리
        
        # 랜덤 시드 설정 (재현 가능한 결과)
        random.seed(42)

    def _adjust_params(self, board: List[List[int]]):
        """게임 단계별 탐색 파라미터 조정"""
        empty = sum(cell == 0 for row in board for cell in row)
        
        # 게임 초반 (16-12 빈칸) - 다양한 가능성 탐색
        if empty >= 12:
            self.exploration = self.exploration_base * 1.4
            self.heuristic_weight = self.heuristic_weight_base * 0.6
            MCTSNode.SIMULATION_DEPTH = 150
        
        # 게임 중반 (11-6 빈칸) - 균형적 탐색
        elif empty >= 6:
            self.exploration = self.exploration_base * 1.2
            self.heuristic_weight = self.heuristic_weight_base * 1.2
            MCTSNode.SIMULATION_DEPTH = 250
        
        # 게임 후반 (5-0 빈칸) - 전술적 탐색
        else:
            self.exploration = self.exploration_base * 0.8
            self.heuristic_weight = self.heuristic_weight_base * 1.6
            MCTSNode.SIMULATION_DEPTH = 350

    def _danger_level(self, pos: Tuple[int,int], piece: Tuple[int,int,int,int]) -> int:
        """위치에 피스를 놓았을 때의 위험도 평가"""
        # 캐싱 키
        cache_key = (pos, piece, tuple(tuple(r) for r in self.board))
        if cache_key in MCTSNode.danger_cache:
            return MCTSNode.danger_cache[cache_key]
        
        # 보드 복사 및 피스 배치
        board_copy = [row.copy() for row in self.board]
        board_copy[pos[0]][pos[1]] = MCTSNode._encode_piece(piece)
        
        danger_level = 0
        
        # 1. 직접적인 승리 가능성 체크
        if MCTSNode._check_win(board_copy):
            danger_level = 10  # 즉시 승리
        else:
            # 2. 양방 3목 체크 (가장 위험)
            fork_count = MCTSNode._check_fork_opportunities(board_copy, pos[0], pos[1], piece)
            if fork_count >= 2:
                danger_level = 8  # 매우 위험 (양방 3목)
            elif fork_count == 1:
                danger_level = 5  # 위험 (한 방향 3목)
            
            # 3. 다음 턴에 상대가 위험한 피스를 얻을 수 있는지
            board_key = tuple(tuple(r) for r in board_copy)
            for p in self.available_pieces:
                if p != piece:
                    risk = MCTSNode._opponent_can_win_cached(board_key, p)
                    if risk == 2:  # 양방 3목 가능
                        danger_level = max(danger_level, 7)
                    elif risk == 1:  # 승리 가능
                        danger_level = max(danger_level, 3)
        
        # 캐싱
        MCTSNode.danger_cache[cache_key] = danger_level
        return danger_level

    def _iterate(self):
        """MCTS 한 번의 반복(선택-확장-시뮬레이션-역전파)"""
        node = self.root
        
        # 1. 선택: 리프 노드나 확장 가능한 노드를 찾을 때까지 트리 탐색
        while not node.untried_actions and node.children:
            node = node.best_child(self.exploration, self.heuristic_weight)
        
        # 2. 확장: 시도하지 않은 액션이 있으면 확장
        if node.untried_actions:
            node = node.expand()
        
        # 3. 시뮬레이션: 무작위 플레이아웃으로 결과 추정
        result = node.simulate()
        
        # 4. 역전파: 결과를 루트까지 전파
        node.backpropagate(result)

    def _search(self):
        """MCTS 메인 탐색 루프"""
        start_time = time.time()
        end_time = start_time + self.MAX_TURN_TIME * 0.95  # 시간 제한의 95%만 사용
        iters = 0
        
        # 병렬 처리 비활성화: 동시성 문제 해결을 위해 순차 실행으로 변경
        # future_results = []
        # num_workers = min(8, os.cpu_count() or 4)
        # batch_size = 16
        
        # with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #     while time.time() < end_time and iters < self.ITERATION_CAP:
        #         new_futures = []
        #         for _ in range(batch_size):
        #             if time.time() >= end_time or iters >= self.ITERATION_CAP:
        #                 break
        #             new_futures.append(executor.submit(self._iterate))
                
        #         future_results.extend(new_futures)
                
        #         done, not_done = wait(
        #             future_results, 
        #             timeout=0.01, 
        #             return_when=FIRST_COMPLETED
        #         )
                
        #         for future in done:
        #             future_results.remove(future)
        #             try:
        #                 future.result()  # 예외 체크
        #                 iters += 1
        #             except Exception as e:
        #                 if self.debug:
        #                     print(f"탐색 오류: {str(e)}")
                
        #         # 일정 간격으로 가장 유망한 자식 노드로 집중 (현재 비활성화 상태 유지)
        #         if iters % 200 == 0 and self.root.children and iters > 0:
        #             best_children = sorted(
        #                 self.root.children, 
        #                 key=lambda c: c.visits, 
        #                 reverse=True
        #             )[:3]
                    
        #             if best_children and best_children[0].visits > self.root.visits * 0.4:
        #                 if self.debug:
        #                     print(f"탐색 집중: {best_children[0].visits}회 방문한 노드로 집중 (현재 비활성화됨)")
        #                 # self.root = best_children[0] # <--- 비활성화 상태 유지

        # 순차 실행 루프
        while time.time() < end_time and iters < self.ITERATION_CAP:
            try:
                self._iterate()
                iters += 1
            except Exception as e:
                if self.debug:
                    print(f"탐색 오류 (순차 실행 중): {str(e)}")
            
            # 일정 간격으로 가장 유망한 자식 노드로 집중 (현재 비활성화 상태 유지)
            # 이 로직은 병렬 처리와 별개이므로, 필요시 활성화 고려 가능 (단, 이전 이슈 재발 가능성 유의)
            if iters % 200 == 0 and self.root.children and iters > 0:
                # ... (탐색 집중 로직은 이전과 동일하게 주석 처리 또는 신중히 관리) ...
                pass # 현재는 탐색 집중 로직 비활성화 유지

        # 탐색 통계 업데이트
        self.last_search_iterations = iters
        self.last_search_time = time.time() - start_time
        
        if self.debug:
            print(f"MCTS 탐색 완료 (순차 실행): {iters}회 반복, {self.last_search_time:.2f}초 소요")

    def _analyze_opponent_trends(self) -> List[Tuple[Optional[int], float]]:
        """상대방 전략 경향 분석"""
        # MBTI 속성별 분포 분석
        attr_distribution = []
        
        for i in range(4):  # 각 속성별 (I/E, N/S, T/F, P/J)
            zeros = sum(1 for r in range(4) for c in range(4) 
                      if self.board[r][c] != 0 and 
                      MCTSNode._decode_piece(self.board[r][c])[i] == 0)
            
            ones = sum(1 for r in range(4) for c in range(4) 
                     if self.board[r][c] != 0 and 
                     MCTSNode._decode_piece(self.board[r][c])[i] == 1)
            
            # 뚜렷한 선호 경향이 있는지 확인
            if zeros > ones + 2:  # 0속성(I/N/T/P) 선호
                attr_distribution.append((0, 0.8))
            elif ones > zeros + 2:  # 1속성(E/S/F/J) 선호
                attr_distribution.append((1, 0.8))
            elif zeros > ones:  # 약한 0속성 선호
                attr_distribution.append((0, 0.4))
            elif ones > zeros:  # 약한 1속성 선호
                attr_distribution.append((1, 0.4))
            else:  # 선호 불명확
                attr_distribution.append((None, 0.0))
        
        return attr_distribution

    def _adjust_children_by_trends(self, attr_distribution: List[Tuple[Optional[int], float]]):
        """상대방 경향에 따른 자식 노드 평가 조정"""
        if not self.root.children:
            return
        
        for child in self.root.children:
            piece = child.selected_piece
            if piece:
                # 상대가 선호하는 속성의 피스에 패널티
                trend_penalty = 0.0
                for i, (preferred, weight) in enumerate(attr_distribution):
                    if preferred is not None and piece[i] == preferred:
                        trend_penalty += 0.15 * weight
                
                # 패널티 적용
                if trend_penalty > 0:
                    child.heuristic -= trend_penalty
                    if self.debug and trend_penalty > 0.2:
                        print(f"상대 선호 속성 패널티: {MCTSNode._get_mbti_name(piece)}, -{trend_penalty:.2f}")

    def _check_fork_opportunities(self, pos: Tuple[int, int], piece: Tuple[int, int, int, int]) -> bool:
        """위치에 피스를 놓았을 때 양방 3목 기회 여부"""
        board_copy = [row.copy() for row in self.board]
        board_copy[pos[0]][pos[1]] = MCTSNode._encode_piece(piece)
        
        # 양방 3목 체크 (2개 이상이면 양방 3목)
        return MCTSNode._check_fork_opportunities(board_copy, pos[0], pos[1], piece) >= 2

    def _find_winning_pattern(self) -> Optional[int]:
        """보드에서 승리에 가까운 패턴 찾기"""
        # 1. 완성 가능한 라인 찾기
        for attr_idx in range(4):  # 각 속성별
            for line_type, indices in [
                ('row', [(r, c) for r in range(4) for c in range(4)]),
                ('col', [(r, c) for c in range(4) for r in range(4)]),
                ('diag', [(i, i) for i in range(4)]),
                ('anti_diag', [(i, 3-i) for i in range(4)])
            ]:
                # 해당 라인의 피스들
                pieces = []
                empty_count = 0
                empty_pos = None
                
                for r, c in indices:
                    if self.board[r][c] == 0:
                        empty_count += 1
                        empty_pos = (r, c)
                    else:
                        pieces.append(MCTSNode._decode_piece(self.board[r][c]))
                
                # 한 칸만 비어있고 나머지가 같은 속성을 가진 경우
                if empty_count == 1 and pieces:
                    if all(p[attr_idx] == pieces[0][attr_idx] for p in pieces):
                        return attr_idx
        
        # 2. 완성 가능한 2x2 블록 찾기
        for r in range(3):
            for c in range(3):
                for attr_idx in range(4):
                    block = [
                        (r, c), (r, c+1),
                        (r+1, c), (r+1, c+1)
                    ]
                    
                    pieces = []
                    empty_count = 0
                    empty_pos = None
                    
                    for pos_r, pos_c in block:
                        if self.board[pos_r][pos_c] == 0:
                            empty_count += 1
                            empty_pos = (pos_r, pos_c)
                        else:
                            pieces.append(MCTSNode._decode_piece(self.board[pos_r][pos_c]))
                    
                    # 한 칸만 비어있고 나머지가 같은 속성을 가진 경우
                    if empty_count == 1 and pieces:
                        if all(p[attr_idx] == pieces[0][attr_idx] for p in pieces):
                            return attr_idx
        
        return None

    def select_piece(self) -> Tuple[int,int,int,int]:
        """상대에게 줄 피스 선택"""
        # 디버깅 모드면 정보 출력
        if self.debug:
            print("\n===== 피스 선택 단계 =====")
            print(f"남은 피스: {len(self.available_pieces)}")
            print(f"보드 상태: {sum(1 for r in range(4) for c in range(4) if self.board[r][c] != 0)}/16칸 배치됨")
        
        # 게임 단계에 따른 파라미터 조정
        self._adjust_params(self.board)
        
        # 상대방 전략 경향 분석
        attr_distribution = self._analyze_opponent_trends()
        
        # 빈 칸 수 계산
        empty = sum(cell == 0 for row in self.board for cell in row)
        
        # 승리 패턴 찾기
        winning_attr = self._find_winning_pattern()
        
        # 초기 게임: 위험도 기반 선택 (MCTS 없이 빠른 계산)
        if empty >= 14:  # 2개 이하 피스만 놓인 상태
            empties = [(r, c) for r, c in product(range(4), range(4)) if self.board[r][c] == 0]
            danger_scores = {}
            
            # 각 피스의 위험도 계산
            for p in self.available_pieces:
                # 기본 위험도
                danger = max(self._danger_level(e, p) for e in empties)
                
                
                # 상대 선호 속성 기반 조정
                for i, (preferred, weight) in enumerate(attr_distribution):
                    if preferred is not None and p[i] == preferred:
                        danger += 1.5 * weight
                
                # 승리 패턴 기반 조정
                if winning_attr is not None and p[winning_attr] == 0:  # 승리 패턴에 맞는 속성
                    danger += 3.0
                
                danger_scores[p] = danger
            
            if self.debug:
                # 위험도 상위 3개 피스 출력
                top_dangers = sorted(danger_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                print("위험한 피스 TOP3:")
                for p, score in top_dangers:
                    print(f"  {MCTSNode._get_mbti_name(p)}: {score:.2f}")
            
            # 가장 안전한 피스 선택
            choice = min(self.available_pieces, key=lambda p: danger_scores[p])
            
            if self.debug:
                print(f"선택된 피스: {MCTSNode._get_mbti_name(choice)} (위험도: {danger_scores[choice]:.2f})")
            
            # 선택 기록
            self.move_history.append(choice)
            return choice
        
        # 중후반 게임: MCTS 탐색
        self._search()
        
        # 자식 노드가 없는 경우 (탐색 실패)
        if not self.root.children:
            if self.debug:
                print("MCTS 탐색 실패, 안전한 피스 직접 선택 (select_piece)")
            
            if not self.available_pieces:
                if self.debug:
                    print("피스 선택 오류: MCTS 실패 후 선택할 피스가 없습니다.")
                # 비상 상황: 기본 피스 반환 (실제 게임에서는 발생하기 어려움)
                fallback_piece = MCTSNode.pieces[0] 
                self.move_history.append(fallback_piece)
                return fallback_piece

            danger_scores = {}
            current_board_key = tuple(tuple(r) for r in self.board)

            for p_candidate in self.available_pieces:
                # 이 p_candidate를 상대에게 주었을 때, 상대가 다음 턴에 이기거나 포크를 만들 수 있는지 평가
                # _opponent_can_win_cached는 (보드 상태, 상대가 받은 말)로 평가.
                # 점수: 0=안전, 1=상대 승리 가능, 2=상대 포크 가능
                danger = MCTSNode._opponent_can_win_cached(current_board_key, p_candidate)
                danger_scores[p_candidate] = danger
            
            if self.debug:
                sorted_dangers = sorted(danger_scores.items(), key=lambda x: x[1])
                print("MCTS 실패 후 피스 위험도 (낮을수록 상대에게 안전한 피스):")
                for p_debug, score_debug in sorted_dangers[:5]:
                    print(f"  {MCTSNode._get_mbti_name(p_debug)}: {score_debug}")
            
            min_danger_score = min(danger_scores.values())
            best_fallback_pieces = [p for p, score in danger_scores.items() if score == min_danger_score]
            
            choice = random.choice(best_fallback_pieces) if best_fallback_pieces else self.available_pieces[0]
            
            if self.debug:
                print(f"MCTS 실패 후 대체 선택 피스: {MCTSNode._get_mbti_name(choice)} (계산된 위험도: {danger_scores.get(choice, 'N/A')})")
            
            self.move_history.append(choice)
            return choice
        
        # 6. 최고의 자식 노드 사용
        best = max(self.root.children, key=lambda c: c.visits)
        self.root = best
        choice = best.selected_piece
        
        if self.debug:
            print(f"MCTS 선택 피스: {MCTSNode._get_mbti_name(choice)} (방문: {best.visits}회, 승률: {best.wins/best.visits:.2f})")
        
        # 선택 기록
        self.move_history.append(choice)
        return choice

    def place_piece(self, piece: Tuple[int,int,int,int]) -> Tuple[int,int]:
        """주어진 피스를 보드에 배치"""
        # 1) MCTS 루트를 'place' 단계로 재초기화
        self.root = MCTSNode(
            board=self.board,
            available_pieces=self.available_pieces,
            player_phase='place',
            selected_piece=piece
        )
        if self.debug:
           print("\n===== 피스 배치 단계 (place) =====")
           print(f"배치할 피스: {MCTSNode._get_mbti_name(piece)}")
        # 1. 즉시 승리 가능한 위치 체크
        for r, c in product(range(4), range(4)):
            if self.board[r][c] == 0:
                # 임시 배치
                temp = [row.copy() for row in self.board]
                temp[r][c] = MCTSNode._encode_piece(piece)
                
                # 승리 체크
                if MCTSNode._check_win(temp):
                    # 승리 위치 기록
                    self.move_history.append((r, c))
                    if self.debug:
                        print(f"즉시 승리 위치 발견: ({r},{c})")
                    return (r, c)

        # 2. 양방 3목 기회 찾기
        fork_opportunities = []
        for r, c in product(range(4), range(4)):
            if self.board[r][c] == 0:
                if self._check_fork_opportunities((r, c), piece):
                    fork_opportunities.append((r, c))
                    
        if fork_opportunities:
            # 양방 3목 위치 중 중앙에 가까운 것 선택
            choice = min(fork_opportunities, 
                       key=lambda pos: abs(pos[0]-1.5) + abs(pos[1]-1.5))
            
            self.move_history.append(choice)
            if self.debug:
                print(f"양방 3목 기회 활용: {choice}")
            return choice
        
        # 3. 위험한 위치 파악
        danger_spots = []
        for r, c in product(range(4), range(4)):
            if self.board[r][c] == 0:
                danger = self._danger_level((r, c), piece)
                if danger > 2:  # 위험도 임계값
                    danger_spots.append(((r, c), danger))
        
        # 위험도 순으로 정렬
        danger_spots.sort(key=lambda x: x[1], reverse=True)
        
        if self.debug and danger_spots:
            print("위험 위치:")
            for (r, c), score in danger_spots[:3]:  # 상위 3개만 출력
                print(f"  ({r},{c}): {score}")
        
        # 4. MCTS 탐색
        self._search()
        
        # 5. 자식 노드가 없는 경우 (탐색 실패)
        if not self.root.children:
            if self.debug:
                print("MCTS 탐색 실패, 안전한 위치 직접 선택")
            
            empties = [(r,c) for r,c in product(range(4), range(4)) 
                      if self.board[r][c] == 0]

            if not empties:
                if self.debug:
                    print("피스 배치 오류: MCTS 실패 후 둘 빈칸이 없습니다. 게임이 이미 종료되었어야 합니다.")
                self.move_history.append((0,0)) 
                return (0,0)

            best_fallback_move = None
            min_opponent_threat_score = float('inf')
            # 'piece'는 P2가 P1에게 준, 현재 P1이 배치해야 할 말입니다.
            encoded_current_piece = MCTSNode._encode_piece(piece) 

            for r_idx, c_idx in empties:
                current_spot_threat_score = 0.0 
                
                # P1이 (r_idx, c_idx)에 'piece'를 놓는다고 시뮬레이션
                self.board[r_idx][c_idx] = encoded_current_piece 
                board_key_after_my_move = tuple(tuple(r) for r in self.board)
                
                # 이제, P1이 이 위치에 말을 놓은 후, P1이 self.available_pieces에서 P2에게 말을 선택해 줄 차례입니다.
                # P2가 그 말을 받아 최적으로 두었을 때의 위협을 평가합니다.
                # 이 휴리스틱은 P1이 P2에게 줄 수 있는 모든 말에 대한 위협의 합계를 사용합니다.
                if not self.available_pieces: # P1이 다음 select_piece 단계에서 P2에게 줄 말이 없는 경우
                    pass # P1이 말을 주는 것으로부터 발생하는 추가 위협은 없음
                else:
                    # P1이 다음 턴에 P2에게 줄 수 있는 각 말에 대해...
                    for piece_p1_might_give_to_p2 in self.available_pieces:
                        # ...P2가 그 말을 받고 board_key_after_my_move 상태에서 최적으로 두었을 때의 위험은?
                        risk = MCTSNode._opponent_can_win_cached(board_key_after_my_move, piece_p1_might_give_to_p2)
                        if risk == 2:  # P2가 이 말을 받으면 포크를 만들 수 있음
                            current_spot_threat_score += 10.0 # 매우 높은 위협도
                        elif risk == 1:  # P2가 이 말을 받으면 승리할 수 있음
                            current_spot_threat_score += 5.0 # 높은 위협도
                
                self.board[r_idx][c_idx] = 0 # 시뮬레이션 되돌리기: P1의 말 배치 취소
                
                # 중앙 선호도 휴리스틱 추가
                centrality_penalty = (abs(r_idx - 1.5) + abs(c_idx - 1.5)) 
                current_spot_threat_score += centrality_penalty * 0.1 # 중앙 가중치는 작게 설정

                if current_spot_threat_score < min_opponent_threat_score:
                    min_opponent_threat_score = current_spot_threat_score
                    best_fallback_move = (r_idx, c_idx)
                elif current_spot_threat_score == min_opponent_threat_score:
                    # 점수가 같으면 중앙에 더 가까운 곳을 선호 (동점 처리)
                    if best_fallback_move is None or \
                       (abs(r_idx - 1.5) + abs(c_idx - 1.5)) < \
                       (abs(best_fallback_move[0] - 1.5) + abs(best_fallback_move[1] - 1.5)):
                        best_fallback_move = (r_idx, c_idx)
            
            # 모든 빈칸을 확인한 후
            if best_fallback_move is not None:
                choice = best_fallback_move
            else:
                # 이 경우는 empties가 비어있지 않다면 발생하기 어렵지만 (min_opponent_threat_score가 갱신되었을 것이므로),
                # 안전장치로 empties가 있다면 그 중 가장 중앙을 선택합니다.
                if empties: 
                    if self.debug:
                        print("MCTS 실패 후 모든 대체 위치가 매우 높은 위협도 (또는 로직 오류), 중앙 우선 선택")
                    empties.sort(key=lambda pos: abs(pos[0]-1.5) + abs(pos[1]-1.5)) # 중앙 우선 정렬
                    choice = empties[0]
                else: 
                    # 이 경우는 블록 시작 부분의 'if not empties:'에서 이미 처리되었어야 합니다.
                    if self.debug:
                        print("피스 배치 오류: MCTS 실패 분석 중 빈 칸 없음 (이중 확인). (0,0) 강제 반환.")
                    choice = (0,0) 

            self.move_history.append(choice)
            if self.debug:
                debug_score_str = f"{min_opponent_threat_score:.2f}" if best_fallback_move is not None and min_opponent_threat_score != float('inf') else 'N/A'
                print(f"MCTS 실패 후 대체 선택 위치: {choice} (계산된 위협 점수: {debug_score_str})")
            return choice
        
        # 6. 최고의 자식 노드 사용
        best_child_node = max(self.root.children, key=lambda c: c.visits)
        self.root = best_child_node # 다음 탐색을 위해 루트를 업데이트
        
        # place_piece는 놓을 위치 (r,c)를 반환해야 함
        # best_child_node.action_taken이 (r,c) 위치임
        choice = best_child_node.action_taken 
        
        if self.debug:
            # MCTSNode._get_mbti_name(choice) 대신 choice (위치)를 직접 출력
            print(f"MCTS 선택 위치: {choice} (방문: {best_child_node.visits}회, 승률: {best_child_node.wins/best_child_node.visits:.2f} (자식노드 기준))")
        
        # 선택 기록 (위치)
        self.move_history.append(choice)
        return choice