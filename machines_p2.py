import numpy as np
import random
from itertools import product

import time
import math
from copy import deepcopy
from typing import List, Tuple, Optional

# 게임 상태를 나타내는 타입 정의
GameState = Tuple[List[List[int]], Tuple]  # (board, available_pieces)

class Node:
    def __init__(self, state: GameState, pieces: List[Tuple], parent=None, move=None, is_opponent_turn=False):
        """
        Node 초기화
        :param state: (board, available_pieces) - make_move 함수로 계산된, 이 노드가 나타내는 게임 상태 (move와 piece가 이미 적용된 상태)
        :param pieces: 전체 게임 조각 정보 (16개)
        :param parent: 부모 노드
        :param move: 이 노드에 도달하기 위해 부모 노드에서 취한 행동 (piece, position) 또는 piece
        :param is_opponent_turn: 이 노드의 상태가 어떤 플레이어의 턴인지 여부
        """
        # 게임 상태 저장 (make_move 함수가 계산한 최종 상태를 그대로 저장)
        self.state = state
        self.board = deepcopy(state[0])  # 전달받은 state의 board를 깊은 복사
        self.available_pieces = tuple(state[1])  # 전달받은 state의 available_pieces를 튜플로 저장
        self.pieces = pieces  # 전체 게임 조각 정보
        
        # 노드 관계 정보
        self.parent = parent
        self.move = move  # 이 노드에 도달하기 위해 취한 행동
        self.children = []
        
        # MCTS 통계
        self.wins = 0
        self.visits = 0
        self.is_opponent_turn = is_opponent_turn
        
        # 평가 속성들
        self.trap_potential = 0.0  # 트랩을 만들 수 있는 잠재력
        self.win_potential = 0.0   # 승리로 이어질 수 있는 잠재력
        self.defense_value = 0.0   # 방어적 가치
        self.opponent_threat = 0.0 # 상대방 위협 수준
        
        # 시도하지 않은 수들 초기화 (Expansion 단계에서 사용)
        self.untried_moves = self._get_possible_moves()
    
    def get_piece_index(self, piece, pieces):
        """조각의 인덱스 반환 (1-based)"""
        return pieces.index(piece) + 1
    
    def _get_possible_moves(self):
        if isinstance(self.move, tuple) and len(self.move) == 2:
            # place_piece 노드: 가능한 모든 위치 반환
            return [(row, col) for row, col in product(range(4), range(4)) 
                   if self.board[row][col] == 0]
        else:
            # select_piece 노드: 가능한 모든 조각 반환
            return list(self.available_pieces)
    
    def add_child(self, move):
        new_board = deepcopy(self.board)
        new_pieces = list(self.available_pieces)
        
        if isinstance(move, tuple) and len(move) == 2:
            # place_piece 노드
            row, col = move
            selected_piece = self.move  # 부모 노드에서 선택된 조각
            new_board[row][col] = selected_piece
        else:
            # select_piece 노드
            new_pieces.remove(move)
            
        child = Node(new_board, new_pieces, self, move, not self.is_opponent_turn)
        self.children.append(child)
        return child
    
    def get_uct_score(self, c_param=1.41):
        """UCT 점수 계산 (평가 속성들을 보너스/패널티로 활용)"""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return float('inf')
        
        # 기본 UCT 점수
        exploitation = self.wins / self.visits
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        uct = exploitation + exploration
        
        # 평가 속성들을 보너스/패널티로 활용
        if self.is_opponent_turn:
            # 상대 턴: 트랩과 승리 잠재력은 부정적, 방어 가치는 긍정적
            bonus = (self.defense_value - self.trap_potential - self.win_potential) * 0.1
        else:
            # 내 턴: 트랩과 승리 잠재력은 긍정적, 상대 위협은 부정적
            bonus = (self.trap_potential + self.win_potential - self.opponent_threat) * 0.1
        
        return uct + bonus

    def is_terminal(self, check_win_func):
        """게임 종료 여부 확인"""
        if check_win_func(self.board, self.pieces):
            return True
        return all(cell != 0 for row in self.board for cell in row)
    
    def is_fully_expanded(self):
        """모든 가능한 수를 시도했는지 확인"""
        return len(self.get_possible_moves()) == len(self.children)

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.time_limit = 0.45  # 각 턴당 0.45초 사용 (여유 시간 확보)
    
    def check_line(self, line, pieces):
        """한 줄의 승리 여부 확인"""
        if 0 in line:  # 빈 칸이 있으면 승리가 아님
            return False
            
        # 각 속성(4차원)에 대해 모든 조각이 같은 값을 가지는지 확인
        line_pieces = [pieces[p-1] for p in line]  # 인덱스를 실제 조각으로 변환
        return any(all(p[dim] == line_pieces[0][dim] for p in line_pieces)
                  for dim in range(4))
    
    def check_2x2_subgrid_win(self, board: List[List[int]], pieces: List[Tuple]) -> bool:
        """2x2 부분 격자의 승리 여부 확인"""
        for r in range(3):  # 0, 1, 2
            for c in range(3):  # 0, 1, 2
                subgrid_indices = [board[r][c], board[r][c+1], 
                                 board[r+1][c], board[r+1][c+1]]
                
                # 2x2 칸이 모두 채워져 있는지 확인
                if 0 not in subgrid_indices:
                    subgrid_pieces = [pieces[idx-1] for idx in subgrid_indices]
                    
                    # 4가지 속성 중 하나라도 모두 같은지 확인
                    for dim in range(4):  # 0:I/E, 1:N/S, 2:T/F, 3:P/J
                        if all(p[dim] == subgrid_pieces[0][dim] for p in subgrid_pieces):
                            return True  # 승리 조건 만족
        
        return False  # 2x2 승리 조건 불만족
    
    def check_win(self, board: List[List[int]], pieces: List[Tuple]) -> bool:
        """승리 조건 검사"""
        # 가로 라인
        for row in board:
            if self.check_line(row, pieces):
                return True
        
        # 세로 라인
        for col in zip(*board):
            if self.check_line(col, pieces):
                return True
        
        # 대각선
        diag1 = [board[i][i] for i in range(4)]
        diag2 = [board[i][3-i] for i in range(4)]
        if self.check_line(diag1, pieces) or self.check_line(diag2, pieces):
            return True
        
        # 2x2 부분 격자 검사
        if self.check_2x2_subgrid_win(board, pieces):
            return True
        
        return False
    
    def evaluate_defense_value(self, node):
        """방어 가치 평가"""
        # TODO: 구현 예정
        pass
    
    def evaluate_opponent_threat(self, node):
        """상대방 위협 수준 평가"""
        # TODO: 구현 예정
        pass
    
    def evaluate_node(self, node):
        """노드의 전략적 가치 평가"""
        node.defense_value = self.evaluate_defense_value(node)
        node.opponent_threat = self.evaluate_opponent_threat(node)
        # 다른 평가 항목들도 추가 예정
    
    def select(self, node):
        """UCT 값이 가장 높은 자식 노드 선택"""
        while not node.is_terminal(self.check_win):
            if not node.is_fully_expanded():
                return node
            node = max(node.children, key=lambda n: n.get_uct_score())
        return node
    
    def expand(self, node: Node) -> Node:
        """
        새로운 자식 노드 확장
        :param node: 확장할 현재 노드
        :return: 새로 생성된 자식 노드
        """
        if node.is_terminal(self.check_win):
            return node
            
        # 1. 시도하지 않은 행동 선택
        possible_moves = node.get_possible_moves()
        tried_moves = set()
        for child in node.children:
            if isinstance(child.move, tuple):
                tried_moves.add(child.move)
            else:
                if child.move in possible_moves:
                    tried_moves.add(child.move)
        
        untried_moves = [move for move in possible_moves if move not in tried_moves]
        if not untried_moves:
            return node
        
        selected_move = np.random.choice(untried_moves)
        
        # 2. make_move 함수로 다음 상태 계산
        next_state = self.make_move(node.get_state(), selected_move, node.pieces)
        
        # 3. 새로운 자식 노드 생성 (계산된 상태 전달)
        child = Node(
            state=next_state,  # make_move가 계산한 다음 상태
            pieces=node.pieces,  # 전체 게임 조각 정보
            parent=node,  # 부모 노드
            move=selected_move,  # 선택된 행동
            is_opponent_turn=not node.is_opponent_turn  # 턴 전환
        )
        
        # 4. 자식 노드 평가 및 추가
        self.evaluate_node(child)  # 노드 평가
        node.children.append(child)
        
        # 5. 새로 생성된 자식 노드 반환
        return child
    
    def simulate(self, node: Node) -> float:
        """
        MCTS Simulation 단계: 현재 노드부터 게임 종료까지 플레이아웃 진행 (휴리스틱 기반)
        :param node: 시뮬레이션을 시작할 노드
        :return: 시뮬레이션 결과 점수 (1.0: 승리, 0.5: 무승부, 0.0: 패배)
        """
        current_board = deepcopy(node.board)
        current_available_pieces = list(node.available_pieces)  # 시뮬레이션 중에는 가변 리스트 사용
        current_player_is_opponent = node.is_opponent_turn  # 현재 턴 플레이어
        
        while not self.check_win(current_board, self.pieces):
            if len(current_available_pieces) == 0:  # 더 이상 둘 수 있는 조각이 없음
                break
                
            if current_player_is_opponent:
                # 상대방 턴 시뮬레이션 (P1처럼 행동)
                piece = current_available_pieces[0]  # 첫 번째 가능한 조각
                move = self.opponent_playout(current_board, piece, self.pieces)
                if move is None:  # 둘 수 있는 위치가 없음
                    break
                selected_move = (piece, move)
            else:
                # AI (P2) 턴 시뮬레이션 (P2처럼 행동)
                piece = self.heuristic_playout(current_board, current_available_pieces, self.pieces)
                selected_move = piece
            
            # make_move 함수로 다음 상태 계산
            current_state = (current_board, current_available_pieces)
            next_state = self.make_move(current_state, selected_move, self.pieces)
            current_board = next_state[0]
            current_available_pieces = list(next_state[1])  # 다음 턴을 위해 리스트로 변환
            
            current_player_is_opponent = not current_player_is_opponent  # 턴 전환
        
        return self.evaluate_simulation_result(current_board, self.pieces, current_player_is_opponent)
    
    def heuristic_playout(self, board: List[List[int]], available_pieces: List[Tuple], 
                         pieces: List[Tuple]) -> Tuple:
        """
        AI (P2)의 휴리스틱 기반 시뮬레이션 - 방어적 성향
        :return: 선택된 조각
        """
        best_piece = None
        best_score = float('-inf')
        
        for piece in available_pieces:
            score = 0
            
            # 1. 상대방의 승리를 방해할 수 있는 조각 선호 (방어 우선)
            score += self.evaluate_blocking_potential(piece, pieces) * 3.0
            
            # 2. 안전한 조각 선호 (리스크 회피)
            score += self.evaluate_safety(piece, board, pieces) * 2.0
            
            # 3. 승리 가능성이 있는 조각도 고려 (기회주의적 공격)
            score += self.evaluate_winning_potential(piece, pieces) * 1.5
            
            # 4. 다양한 속성을 가진 조각 선호 (유연성)
            score += sum(piece) * 0.5
            
            if score > best_score:
                best_score = score
                best_piece = piece
        
        return best_piece if best_piece else np.random.choice(available_pieces)
    
    def opponent_playout(self, board: List[List[int]], piece: Tuple, 
                        pieces: List[Tuple]) -> Optional[Tuple[int, int]]:
        """
        상대방(P1)의 휴리스틱 기반 시뮬레이션 - 공격적 성향
        :return: 선택된 위치 (row, col) 또는 None
        """
        available_locs = [(r, c) for r, c in product(range(4), range(4)) 
                         if board[r][c] == 0]
        if not available_locs:
            return None
            
        best_move = None
        best_score = float('-inf')
        
        for row, col in available_locs:
            score = 0
            
            # 1. 즉시 승리 가능한 위치 선호 (기회가 있다면 승리)
            temp_board = deepcopy(board)
            temp_board[row][col] = self.get_piece_index(piece, pieces)
            if self.check_win(temp_board, pieces):
                return (row, col)
            
            # 2. 트랩을 만들 수 있는 위치 선호 (전략적 공격)
            score += self.evaluate_trap_potential_at(row, col, piece, board, pieces) * 2.0
            
            # 3. 중앙에 가까운 위치 선호 (전략적 중요성)
            center_dist = abs(row - 1.5) + abs(col - 1.5)  # 중앙으로부터의 맨해튼 거리
            score += (4 - center_dist) * 0.5
            
            # 4. 라인 완성 가능성이 높은 위치 선호
            score += self.evaluate_line_completion(row, col, piece, board, pieces)
            
            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        return best_move if best_move else available_locs[np.random.randint(len(available_locs))]
    
    def evaluate_simulation_result(self, board: List[List[int]], pieces: List[Tuple],
                                 is_opponent_turn: bool) -> float:
        """
        시뮬레이션 결과 평가
        :return: AI(P2) 관점에서의 점수 (1.0: 승리, 0.5: 무승부, 0.0: 패배)
        """
        if self.check_win(board, pieces):
            return 0.0 if is_opponent_turn else 1.0  # 마지막 수를 둔 플레이어의 승리
        return 0.5  # 무승부
    
    def evaluate_winning_potential(self, piece: Tuple, pieces: List[Tuple]) -> float:
        """조각의 승리 잠재력 평가"""
        # TODO: 구현 예정
        return 0.0
    
    def evaluate_blocking_potential(self, piece: Tuple, pieces: List[Tuple]) -> float:
        """조각의 방어 잠재력 평가"""
        # TODO: 구현 예정
        return 0.0
    
    def evaluate_safety(self, piece: Tuple, board: List[List[int]], 
                       pieces: List[Tuple]) -> float:
        """조각의 안전성 평가"""
        # TODO: 구현 예정
        return 0.0
    
    def evaluate_trap_potential_at(self, row: int, col: int, piece: Tuple, 
                                 board: List[List[int]], pieces: List[Tuple]) -> float:
        """특정 위치의 트랩 생성 잠재력 평가"""
        # TODO: 구현 예정
        return 0.0
    
    def evaluate_line_completion(self, row: int, col: int, piece: Tuple, 
                               board: List[List[int]], pieces: List[Tuple]) -> float:
        """라인 완성 가능성 평가"""
        # TODO: 구현 예정
        return 0.0
    
    def backpropagate(self, node, result):
        """결과를 부모 노드들에게 전파"""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1 - result  # 결과 반전 (min-max)
    
    def get_best_move(self, root, is_selection=True):
        """MCTS 실행하여 최선의 수 선택"""
        start_time = time.time()
        
        while time.time() - start_time < self.time_limit:
            node = self.select(root)
            node = self.expand(node)
            result = self.simulate(node)
            self.backpropagate(node, result)
        
        if is_selection:
            # select_piece: 승률이 가장 낮은 수 선택 (상대방에게 불리한 조각)
            best_child = min(root.children, key=lambda n: n.wins/n.visits if n.visits > 0 else float('inf'))
        else:
            # place_piece: 승률이 가장 높은 수 선택
            best_child = max(root.children, key=lambda n: n.wins/n.visits if n.visits > 0 else float('-inf'))
        
        return best_child.move
    
    def select_piece(self):
        """상대방에게 줄 조각 선택"""
        root = Node(self.board, self.available_pieces, pieces=self.pieces, is_opponent_turn=True)
        return self.get_best_move(root, is_selection=True)
    
    def place_piece(self, selected_piece):
        """조각을 놓을 위치 선택"""
        root = Node(self.board, [selected_piece], self.pieces, 
                   move=selected_piece, is_opponent_turn=False)
        return self.get_best_move(root, is_selection=False)