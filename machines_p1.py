import numpy as np
from itertools import product
import time
import math
from copy import deepcopy
from typing import Tuple, List, Optional

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
    
    def get_state(self) -> GameState:
        """현재 게임 상태 반환"""
        return (self.board, self.available_pieces)
    
    def is_terminal(self, check_win_func) -> bool:
        """게임 종료 여부 확인"""
        if check_win_func(self.board, self.pieces):
            return True
        return all(cell != 0 for row in self.board for cell in row)
    
    def is_fully_expanded(self) -> bool:
        """모든 가능한 수를 시도했는지 확인"""
        return len(self.get_possible_moves()) == len(self.children)
    
    def get_possible_moves(self):
        """현재 노드에서 가능한 모든 수를 반환"""
        if isinstance(self.move, tuple) and len(self.move) == 2:
            # place_piece 노드: 가능한 모든 위치 반환
            return [(row, col) for row, col in product(range(4), range(4)) 
                   if self.board[row][col] == 0]
        else:
            # select_piece 노드: 가능한 모든 조각 반환
            return list(self.available_pieces)
    
    def get_uct_score(self, c_param=1.41, heuristic_weight=0.1) -> float:
        """UCT 점수 계산 (휴리스틱 평가 속성 반영)"""
        if self.visits == 0:
            # 방문하지 않은 노드는 무한대 점수를 주어 우선 탐색하도록 함
            return float('inf')
        if self.parent is None:
            # 루트 노드는 자식 중 하나를 선택해야 하므로, 루트 자체의 UCT 점수는 의미 없음 (또는 무한대)
            return float('inf') 

        # 1. Exploitation Term (평균 승률)
        # self.wins는 항상 P1의 관점에서 기록된 승수라고 가정 (backpropagate에서 조정)
        exploitation = self.wins / self.visits

        # 2. Exploration Term (탐험 가중치)
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)

        # 3. Heuristic Bonus Term (평가 속성 기반 보너스/페널티)
        # evaluate_node에서 계산된 값들을 P1의 관점에서 UCT 점수에 반영
        # win_potential: P1 승리 잠재력 (높을수록 좋음)
        # trap_potential: P1 트랩 잠재력 (높을수록 좋음)
        # defense_value: P1 방어 가치 (P1 안전성, 높을수록 좋음. P2의 방어 성공도는 음수로 저장됨)
        # opponent_threat: 상대(P2) 위협 (낮을수록 좋음 -> 빼준다)
        # 이 공식은 evaluate_node에서 각 속성의 부호를 P1의 유불리에 맞게 설정했다는 가정하에 동작
        heuristic_bonus = (self.win_potential + self.trap_potential + self.defense_value - self.opponent_threat) * heuristic_weight

        # 최종 UCT 점수
        uct_score = exploitation + exploration + heuristic_bonus
        
        return uct_score

class P1:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) 
                      for k in range(2) for l in range(2)]  # 전체 16개 조각
        self.board = board
        self.available_pieces = available_pieces
        self.time_limit = 0.45  # 각 턴당 0.45초 사용
    
    def get_piece_from_index(self, index: int, pieces: List[Tuple]) -> Optional[Tuple]:
        """인덱스로부터 조각 정보 반환"""
        if index == 0:  # 빈 칸
            return None
        return pieces[index - 1]  # 1-based 인덱스
    
    def get_piece_index(self, piece: Tuple, pieces: List[Tuple]) -> int:
        """조각의 인덱스 반환 (1-based)"""
        return pieces.index(piece) + 1
    
    def make_move(self, state: GameState, move, pieces: List[Tuple]) -> GameState:
        """
        현재 상태에서 주어진 행동을 취했을 때의 다음 상태 반환
        :param state: (board, available_pieces) - 현재 게임 상태
        :param move: (piece, position) 또는 piece
        :param pieces: 전체 게임 조각 정보
        :return: (new_board, new_available_pieces) - 다음 게임 상태
        """
        board, available_pieces = state
        new_board = deepcopy(board)
        new_available_pieces = list(available_pieces)
        
        if isinstance(move, tuple) and len(move) == 2:
            # place_piece: 조각을 보드에 배치
            piece, (row, col) = move
            new_board[row][col] = self.get_piece_index(piece, pieces)
        else:
            # select_piece: 조각을 사용 가능 목록에서 제거
            piece = move
            new_available_pieces.remove(piece)
        
        return (new_board, tuple(new_available_pieces))
    
    def check_line_attributes(self, attributes: List[Tuple], dim: int) -> bool:
        """한 줄의 특정 속성이 모두 같은지 확인"""
        return all(attr[dim] == attributes[0][dim] for attr in attributes)
    
    def check_line(self, line: List[int], pieces: List[Tuple]) -> bool:
        """한 줄의 승리 여부 확인"""
        if 0 in line:  # 빈 칸이 있으면 승리가 아님
            return False
            
        # 각 속성(4차원)에 대해 모든 조각이 같은 값을 가지는지 확인
        line_pieces = [self.get_piece_from_index(idx, pieces) for idx in line]
        return any(self.check_line_attributes(line_pieces, dim) for dim in range(4))
    
    def check_2x2_subgrid_win(self, board: List[List[int]], pieces: List[Tuple]) -> bool:
        """2x2 부분 격자의 승리 여부 확인"""
        for r in range(3):  # 0, 1, 2
            for c in range(3):  # 0, 1, 2
                subgrid_indices = [board[r][c], board[r][c+1], 
                                 board[r+1][c], board[r+1][c+1]]
                
                # 2x2 칸이 모두 채워져 있는지 확인
                if 0 not in subgrid_indices:
                    subgrid_pieces = [self.get_piece_from_index(idx, pieces) for idx in subgrid_indices]
                    
                    # 4가지 속성 중 하나라도 모두 같은지 확인
                    for dim in range(4):  # 0:I/E, 1:N/S, 2:T/F, 3:P/J
                        if self.check_line_attributes(subgrid_pieces, dim):
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
    
    def evaluate_node(self, node: Node) -> None:
        """노드의 전략적 가치 평가. 이 노드는 'expand' 단계에서 방금 생성된 자식 노드임."""
        action_that_led_to_node = node.move
        board_state_of_node = node.board
        all_game_pieces = node.pieces
        # 이 노드 상태에서 다음 플레이어가 선택/사용할 수 있는 조각들
        current_available_pieces_for_node = list(node.available_pieces)

        # 평가 속성 초기화
        node.win_potential = 0.0
        node.trap_potential = 0.0
        node.defense_value = 0.0
        node.opponent_threat = 0.0

        # node.is_opponent_turn이 True라는 것은, 이 노드에서 P2가 행동할 차례라는 의미.
        # 즉, 이전 액션(action_that_led_to_node)은 P1이 수행한 것.
        if node.is_opponent_turn:
            # Case 1: P1이 조각을 *선택*하여 P2에게 주었고, 이 노드는 그 결과 상태. (P2가 이 조각을 놓을 차례)
            if not isinstance(action_that_led_to_node, tuple) and action_that_led_to_node is not None:
                piece_selected_by_p1 = action_that_led_to_node
                # P1의 트랩 잠재력 (P1 관점): 이 조각을 줌으로써 P1이 트랩을 만들었는가? (UCT 보너스)
                node.trap_potential = self.evaluate_trap_potential(piece_selected_by_p1, board_state_of_node, all_game_pieces)
                # 상대(P2)의 위협 (P1 관점): P2가 이 조각으로 얼마나 위협적인 수를 둘 수 있는가? (UCT 페널티)
                node.opponent_threat = self.evaluate_winning_potential(piece_selected_by_p1, board_state_of_node, all_game_pieces)
                # P1의 방어 가치 (P1 관점): P1이 준 조각이 안전한가? (-blocking_potential 값, UCT 보너스)
                node.defense_value = -self.evaluate_blocking_potential(piece_selected_by_p1, board_state_of_node, all_game_pieces)
                node.win_potential = 0.0 # P1이 직접 놓은 것이 아니므로 0

            # Case 2: P1이 조각을 특정 위치에 *놓았고*, 이 노드는 그 결과 상태. (P2가 조각을 선택할 차례)
            elif isinstance(action_that_led_to_node, tuple) and len(action_that_led_to_node) == 2:
                piece_placed_by_p1, (r, c) = action_that_led_to_node
                # P1의 승리 잠재력 (P1 관점): P1이 방금 놓은 수로 라인이 얼마나 완성되었는가? (UCT 보너스)
                node.win_potential = self.evaluate_line_completion(r, c, piece_placed_by_p1, board_state_of_node, all_game_pieces)
                # P2가 조각 선택 단계이므로 P2의 즉각적인 보드 위협 없음.
                node.opponent_threat = 0.0
                # P1이 방금 두었으므로 트랩/방어 가치는 다음 수에 따라 결정 (단순화).
                node.trap_potential = 0.0
                node.defense_value = 0.0

        # node.is_opponent_turn이 False라는 것은, 이 노드에서 P1이 행동할 차례라는 의미.
        # 즉, 이전 액션(action_that_led_to_node)은 P2가 수행한 것.
        else:
            # Case 3: P2가 조각을 특정 위치에 *놓았고*, 이 노드는 그 결과 상태. (P1이 조각을 선택할 차례)
            if isinstance(action_that_led_to_node, tuple) and len(action_that_led_to_node) == 2:
                piece_placed_by_p2, (r_p2, c_p2) = action_that_led_to_node
                # 상대(P2)의 위협 (P1 관점): P2가 놓은 수의 위협도 (P2 라인완성도, UCT 페널티)
                node.opponent_threat = self.evaluate_line_completion(r_p2, c_p2, piece_placed_by_p2, board_state_of_node, all_game_pieces)
                # P1의 방어 가치 (P1 관점): P2의 수가 P1의 위협을 얼마나 막았는가?
                p2_defense_success_score = self.evaluate_defensive_value(r_p2, c_p2, piece_placed_by_p2, board_state_of_node, all_game_pieces, current_available_pieces_for_node)
                # P2가 잘 막을수록 P1에겐 안좋은 상태. UCT에서는 이 값을 P1의 방어 가치로 사용 (보너스). 부호 조정.
                node.defense_value = -p2_defense_success_score 
                # P1이 조각 선택 단계이므로 win/trap 잠재력은 0.
                node.win_potential = 0.0
                node.trap_potential = 0.0
            
            # Case 4: P2가 조각을 *선택*하여 P1에게 주었고, 이 노드는 그 결과 상태. (P1이 이 조각을 놓을 차례)
            elif not isinstance(action_that_led_to_node, tuple) and action_that_led_to_node is not None:
                piece_given_to_p1_by_p2 = action_that_led_to_node
                # P1의 승리 잠재력 (P1 관점): P1이 이 조각으로 얼마나 좋은 수를 둘 수 있는가? (UCT 보너스)
                node.win_potential = self.evaluate_winning_potential(piece_given_to_p1_by_p2, board_state_of_node, all_game_pieces)
                # P1의 트랩 잠재력 (P1 관점): P1이 이 조각으로 트랩을 만들 수 있는가? (UCT 보너스)
                node.trap_potential = self.evaluate_trap_potential(piece_given_to_p1_by_p2, board_state_of_node, all_game_pieces)
                # P1의 방어 가치 (P1 관점): P2가 준 조각이 얼마나 안전한가? (-P2가 준 조각의 공격성, UCT 보너스)
                node.defense_value = -self.evaluate_winning_potential(piece_given_to_p1_by_p2, board_state_of_node, all_game_pieces)
                # P1이 놓을 차례이므로 P2의 즉각적인 위협 없음.
                node.opponent_threat = 0.0

    def evaluate_defensive_value(self, r_p2: int, c_p2: int, piece_p2_placed: Tuple,
                                 board_before_p2_action: List[List[int]],
                                 all_game_pieces: List[Tuple],
                                 p1_available_pieces_for_next_turn: List[Tuple]) -> float:
        """
        P2가 (r_p2, c_p2)에 piece_p2_placed를 놓은 행동의 방어적 가치를 평가.
        이 행동이 P1의 잠재적 위협(승리, 3목, 2x2의 3개 이상)을 얼마나 잘 차단했는지 평가.
        점수가 높을수록 P2의 방어가 성공적이었다는 의미 (P1에게는 불리한 상황 초래).
        """
        defense_score_by_p2 = 0.0
        
        board_after_p2_action = deepcopy(board_before_p2_action)
        if board_after_p2_action[r_p2][c_p2] == 0:
            board_after_p2_action[r_p2][c_p2] = self.get_piece_index(piece_p2_placed, all_game_pieces)
        else:
            return 0.0 

        for p1_next_piece in p1_available_pieces_for_next_turn:
            hypothetical_p1_board = deepcopy(board_before_p2_action)
            if hypothetical_p1_board[r_p2][c_p2] == 0:
                hypothetical_p1_board[r_p2][c_p2] = self.get_piece_index(p1_next_piece, all_game_pieces)

                # 1. P1의 잠재적 라인 위협 (승리 또는 3목) 및 P2의 차단 평가
                lines_at_rc = [
                    hypothetical_p1_board[r_p2], 
                    [hypothetical_p1_board[i][c_p2] for i in range(4)],
                    [hypothetical_p1_board[i][i] for i in range(4)] if r_p2 == c_p2 else [],
                    [hypothetical_p1_board[i][3-i] for i in range(4)] if r_p2 + c_p2 == 3 else []
                ]
                p1_line_threat_blocked = False
                for line_p1_formed in lines_at_rc:
                    if not line_p1_formed: continue
                    for dim in range(4):
                        if 0 not in line_p1_formed: # P1이 놓아서 라인이 꽉 참 (4개)
                            pieces_in_hypo_line = [self.get_piece_from_index(idx, all_game_pieces) for idx in line_p1_formed]
                            # `.attribute` 접근 오류 수정: `p1_next_piece[dim]` 사용
                            if all(p[dim] == p1_next_piece[dim] for p in pieces_in_hypo_line): # P1 승리 라인
                                # `.attribute` 접근 오류 수정: `piece_p2_placed[dim]` 사용
                                if piece_p2_placed[dim] != p1_next_piece[dim]:
                                    defense_score_by_p2 += 3.0
                                    p1_line_threat_blocked = True; break
                        else: # P1이 놓아서 라인이 꽉 차지 않음 (3목 등)
                            pieces_in_hypo_line = [self.get_piece_from_index(idx, all_game_pieces) for idx in line_p1_formed if idx !=0]
                            if len(pieces_in_hypo_line) == 3:
                                # `.attribute` 접근 오류 수정: `p1_next_piece[dim]` 사용
                                count_p1_attr = sum(1 for p in pieces_in_hypo_line if p[dim] == p1_next_piece[dim])
                                if count_p1_attr == 3: # P1이 3목 형성
                                    # `.attribute` 접근 오류 수정: `piece_p2_placed[dim]` 사용
                                    if piece_p2_placed[dim] != p1_next_piece[dim]:
                                        defense_score_by_p2 += 1.5
                                        p1_line_threat_blocked = True; break
                    if p1_line_threat_blocked: break
                if p1_line_threat_blocked: continue 

                # 2. P1의 잠재적 2x2 위협 (3개 또는 4개 완성) 및 P2의 차단 평가
                p1_2x2_threat_blocked = False
                for r_offset in [-1, 0]:
                    for c_offset in [-1, 0]:
                        r_start, c_start = r_p2 + r_offset, c_p2 + c_offset
                        if not (0 <= r_start <= 2 and 0 <= c_start <= 2): continue

                        hypo_subgrid_indices = [
                            hypothetical_p1_board[r_start][c_start], hypothetical_p1_board[r_start+1][c_start],
                            hypothetical_p1_board[r_start][c_start+1], hypothetical_p1_board[r_start+1][c_start+1]
                        ]
                        if 0 in hypo_subgrid_indices: continue
                        
                        for dim_2x2 in range(4):
                            hypo_subgrid_pieces = [self.get_piece_from_index(idx, all_game_pieces) for idx in hypo_subgrid_indices]
                            # `.attribute` 접근 오류 수정: `p1_next_piece[dim_2x2]` 사용
                            count_p1_attr_2x2 = sum(1 for p in hypo_subgrid_pieces if p[dim_2x2] == p1_next_piece[dim_2x2])
                            if count_p1_attr_2x2 >= 3: # P1이 2x2에서 3개 이상 완성 (위협)
                                # `.attribute` 접근 오류 수정: `piece_p2_placed[dim_2x2]` 사용
                                if piece_p2_placed[dim_2x2] != p1_next_piece[dim_2x2]:
                                    defense_score_by_p2 += (2.0 if count_p1_attr_2x2 == 4 else 1.0)
                                    p1_2x2_threat_blocked = True; break
                        if p1_2x2_threat_blocked: break
                    if p1_2x2_threat_blocked: break
                if p1_2x2_threat_blocked: continue 
        return defense_score_by_p2

    def select(self, node: Node) -> Node:
        """UCT 값이 가장 높은 자식 노드 선택"""
        while not node.is_terminal(lambda b, p: self.check_win(b, p)):
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
        if node.is_terminal(lambda b, p: self.check_win(b, p)):
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
        
        while not self.check_win(current_board, node.pieces):
            if len(current_available_pieces) == 0:  # 더 이상 둘 수 있는 조각이 없음
                break
                
            if current_player_is_opponent:
                # 상대방 턴 시뮬레이션 (P2처럼 행동)
                piece = current_available_pieces[0]  # 첫 번째 가능한 조각
                move = self.opponent_playout(current_board, piece, node.pieces)
                if move is None:  # 둘 수 있는 위치가 없음
                    break
                selected_move = (piece, move)
            else:
                # AI (P1) 턴 시뮬레이션 (P1처럼 행동)
                piece = self.heuristic_playout(current_board, current_available_pieces, node.pieces)
                selected_move = piece
            
            # make_move 함수로 다음 상태 계산
            current_state = (current_board, current_available_pieces)
            next_state = self.make_move(current_state, selected_move, node.pieces)
            current_board = next_state[0]
            current_available_pieces = list(next_state[1])  # 다음 턴을 위해 리스트로 변환
            
            current_player_is_opponent = not current_player_is_opponent  # 턴 전환
        
        return self.evaluate_simulation_result(current_board, node.pieces, current_player_is_opponent)
    
    def heuristic_playout(self, board: List[List[int]], available_pieces: List[Tuple], 
                         pieces: List[Tuple]) -> Tuple:
        """
        AI (P1)의 휴리스틱 기반 시뮬레이션 - 공격적 성향
        :return: 선택된 조각
        """
        best_piece = None
        best_score = float('-inf')
        
        for piece in available_pieces:
            score = 0.0
            
            # 1. 승리 가능성이 높은 조각 선호 (공격 우선)
            score += self.evaluate_winning_potential(piece, board, pieces) * 3.0
            
            # 2. 트랩을 만들 수 있는 조각 선호 (전략적 공격)
            score += self.evaluate_trap_potential(piece, board, pieces) * 2.0
            
            # 3. 상대방에게 불리한 조각 선호 (evaluate_blocking_potential의 점수가 낮은 조각)
            score -= self.evaluate_blocking_potential(piece, board, pieces) * 1.5 
            
            # 4. 다양한 속성을 가진 조각 선호 (유연성)
            score += sum(piece) * 0.5
            
            if score > best_score:
                best_score = score
                best_piece = piece
        
        return best_piece if best_piece else np.random.choice(available_pieces)
    
    def opponent_playout(self, board: List[List[int]], piece: Tuple, 
                        pieces: List[Tuple]) -> Optional[Tuple[int, int]]:
        """
        상대방(P2)의 휴리스틱 기반 시뮬레이션 - 방어적 성향
        :return: 선택된 위치 (row, col) 또는 None
        """
        available_locs = [(r, c) for r, c in product(range(4), range(4)) 
                         if board[r][c] == 0]
        if not available_locs:
            return None
            
        best_move = None
        best_score = float('-inf')
        
        for row, col in available_locs:
            score = 0.0
            
            # 1. 즉시 승리 가능한 위치 선호 (기회가 있다면 승리)
            temp_board = deepcopy(board)
            temp_board[row][col] = self.get_piece_index(piece, pieces)
            if self.check_win(temp_board, pieces):
                return (row, col)
            
            # 2. AI의 승리 기회를 차단하는 위치 선호 (방어 우선)
            score += self.evaluate_defensive_value(row, col, piece, board, pieces) * 2.0
            
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
        :return: AI(P1) 관점에서의 점수 (1.0: 승리, 0.5: 무승부, 0.0: 패배)
        """
        if self.check_win(board, pieces):
            return 0.0 if is_opponent_turn else 1.0  # 마지막 수를 둔 플레이어의 승리
        return 0.5  # 무승부
    
    def count_same_attribute_pieces(self, line: List[int], pieces: List[Tuple], 
                                   target_dim: int) -> int:
        """한 줄에서 특정 속성이 같은 조각의 총 개수를 반환"""
        if 0 in line:  # 빈 칸이 있으면 해당 라인은 완성되지 않음
            filled_pieces = [self.get_piece_from_index(idx, pieces) for idx in line if idx != 0]
            if not filled_pieces:  # 모두 빈칸이면 0 반환
                return 0
            # 가장 많이 나타나는 속성 값의 개수를 반환
            values = [p[target_dim] for p in filled_pieces]
            return max(values.count(0), values.count(1))
        return sum(1 for p in [self.get_piece_from_index(idx, pieces) for idx in line]
                  if p[target_dim] == pieces[line[0]-1][target_dim])

    def evaluate_winning_potential(self, piece: Tuple, board: List[List[int]], 
                                 pieces: List[Tuple]) -> float:
        """
        조각의 승리 잠재력 평가
        이 조각을 놓을 수 있는 각 위치에서 새로운 3목/4목이 만들어질 가능성 평가
        """
        score = 0.0
        
        # 1. 각 빈칸에 이 조각을 놓아보고 평가
        for r in range(4):
            for c in range(4):
                if board[r][c] != 0:  # 이미 채워진 칸은 건너뛰기
                    continue
                    
                # 임시로 조각을 놓아보기
                temp_board = deepcopy(board)
                temp_board[r][c] = self.get_piece_index(piece, pieces)
                
                # 2. 이 위치를 지나는 모든 라인 평가
                lines_to_check = [
                    temp_board[r],  # 가로
                    [temp_board[i][c] for i in range(4)],  # 세로
                    [temp_board[i][i] for i in range(4)] if r == c else [],  # 주대각선
                    [temp_board[i][3-i] for i in range(4)] if r + c == 3 else []  # 부대각선
                ]
                
                # 3. 각 라인에서 같은 속성을 가진 조각 수 평가
                for line in lines_to_check:
                    if not line:  # 대각선이 아닌 경우 건너뛰기
                        continue
                    for dim in range(4):
                        count = self.count_same_attribute_pieces(line, pieces, dim)
                        if count == 4:  # 승리
                            score += 10.0
                        elif count == 3:  # 잠재적 승리
                            score += 3.0
                        elif count == 2:  # 발전 가능성
                            score += 0.5
                
                # 4. 2x2 부분 격자 평가
                if r < 3 and c < 3:
                    subgrid = [temp_board[r][c], temp_board[r][c+1],
                              temp_board[r+1][c], temp_board[r+1][c+1]]
                    for dim in range(4):
                        count = self.count_same_attribute_pieces(subgrid, pieces, dim)
                        if count == 4:  # 2x2 승리
                            score += 10.0
                        elif count == 3:  # 잠재적 2x2
                            score += 3.0
        
        return score

    def evaluate_blocking_potential(self, piece: Tuple, board: List[List[int]], 
                                  pieces: List[Tuple]) -> float:
        """
        조각의 방어 잠재력 평가 (P1의 select_piece 관점)
        상대방에게 이 조각을 주었을 때, 상대가 이 조각으로 P1의 승리를 얼마나 잘 막을 수 있는지,
        또는 자신의 승리를 만들 수 있는지를 평가 (낮은 점수가 P1에게 유리)
        """
        score = 0.0
        
        # 1. 상대방이 이 조각으로 P1의 3목을 막을 수 있는 경우 평가
        for r in range(4):
            for c in range(4):
                if board[r][c] != 0:
                    continue
                
                # 임시로 조각을 놓아보기
                temp_board = deepcopy(board)
                temp_board[r][c] = self.get_piece_index(piece, pieces)
                
                # 2. P1의 3목 라인을 차단하는지 확인
                lines_to_check = [
                    (temp_board[r], [(r,i) for i in range(4)]),  # 가로
                    ([temp_board[i][c] for i in range(4)], [(i,c) for i in range(4)]),  # 세로
                    ([temp_board[i][i] for i in range(4)], [(i,i) for i in range(4)]) if r == c else ([], []),  # 주대각선
                    ([temp_board[i][3-i] for i in range(4)], [(i,3-i) for i in range(4)]) if r + c == 3 else ([], [])  # 부대각선
                ]
                
                for line, positions in lines_to_check:
                    if not line:
                        continue
                    for dim in range(4):
                        # P1의 3목을 차단
                        if self.count_same_attribute_pieces(line, pieces, dim) >= 3:
                            score += 2.0
                        # 상대방의 3목 생성
                        count_after = self.count_same_attribute_pieces(line, pieces, dim)
                        if count_after >= 3:
                            score += 3.0
                
                # 3. 2x2 평가
                if r < 3 and c < 3:
                    subgrid = [temp_board[r][c], temp_board[r][c+1],
                              temp_board[r+1][c], temp_board[r+1][c+1]]
                    for dim in range(4):
                        # P1의 2x2를 차단
                        if self.count_same_attribute_pieces(subgrid, pieces, dim) >= 3:
                            score += 2.0
                        # 상대방의 2x2 생성
                        count_after = self.count_same_attribute_pieces(subgrid, pieces, dim)
                        if count_after >= 3:
                            score += 3.0
        
        return score  # 높은 점수 = 상대방에게 유리한 조각 = P1이 주기 싫은 조각

    def evaluate_trap_potential(self, piece: Tuple, board: List[List[int]], 
                              pieces: List[Tuple]) -> float:
        """
        조각의 트랩 잠재력 평가 (P1의 select_piece 관점)
        상대방에게 이 조각을 주었을 때, 상대가 어디에 놓더라도
        다음 턴에 P1이 승리할 수 있는 상황을 만들 수 있는지 평가
        """
        score = 0.0
        available_pieces = list(self.available_pieces)
        if piece in available_pieces:
            available_pieces.remove(piece)
        
        # 1. 상대방이 이 조각을 놓을 수 있는 각 위치 시도
        for r in range(4):
            for c in range(4):
                if board[r][c] != 0:
                    continue
                
                # 상대방이 이 위치에 조각을 놓았다고 가정
                temp_board = deepcopy(board)
                temp_board[r][c] = self.get_piece_index(piece, pieces)
                
                # 2. P1의 다음 턴에서 사용 가능한 각 조각으로
                for next_piece in available_pieces:
                    can_win = False
                    # 3. 놓을 수 있는 각 위치에 대해
                    for next_r in range(4):
                        for next_c in range(4):
                            if temp_board[next_r][next_c] != 0:
                                continue
                            
                            # 4. P1이 승리할 수 있는지 확인
                            next_board = deepcopy(temp_board)
                            next_board[next_r][next_c] = self.get_piece_index(next_piece, pieces)
                            if self.check_win(next_board, pieces):
                                can_win = True
                                score += 5.0  # 트랩 상황 발견
                                break
                        if can_win:
                            break
        
        return score

    def evaluate_trap_potential_at(self, row: int, col: int, piece: Tuple,
                                 board: List[List[int]], pieces: List[Tuple]) -> float:
        """특정 위치에서의 트랩 잠재력 평가 (opponent_playout에서 사용)"""
        score = 0.0
        temp_board = deepcopy(board)
        temp_board[row][col] = self.get_piece_index(piece, pieces)
        
        # 1. 이 위치에 조각을 놓은 후 생기는 3목 상황 평가
        lines_to_check = [
            (temp_board[row], [(row,i) for i in range(4)]),  # 가로
            ([temp_board[i][col] for i in range(4)], [(i,col) for i in range(4)]),  # 세로
            ([temp_board[i][i] for i in range(4)], [(i,i) for i in range(4)]) if row == col else ([], []),  # 주대각선
            ([temp_board[i][3-i] for i in range(4)], [(i,3-i) for i in range(4)]) if row + col == 3 else ([], [])  # 부대각선
        ]
        
        for line, positions in lines_to_check:
            if not line:
                continue
            for dim in range(4):
                count = self.count_same_attribute_pieces(line, pieces, dim)
                if count == 3:  # 3목 상황
                    score += 2.0
                elif count == 2:  # 2목 상황
                    score += 0.5
        
        # 2. 2x2 평가
        if row < 3 and col < 3:
            subgrid = [temp_board[row][col], temp_board[row][col+1],
                      temp_board[row+1][col], temp_board[row+1][col+1]]
            for dim in range(4):
                count = self.count_same_attribute_pieces(subgrid, pieces, dim)
                if count == 3:  # 3개 모임
                    score += 2.0
                elif count == 2:  # 2개 모임
                    score += 0.5
        
        return score

    def evaluate_line_completion(self, row: int, col: int, piece: Tuple,
                               board: List[List[int]], pieces: List[Tuple]) -> float:
        """
        라인 완성 가능성 평가 (opponent_playout에서 사용)
        특정 위치에 조각을 놓았을 때 완성되거나 거의 완성되는 라인의 가치 평가
        """
        score = 0.0
        temp_board = deepcopy(board)
        temp_board[row][col] = self.get_piece_index(piece, pieces)
        
        # 1. 가로/세로 라인 평가
        lines_to_check = [
            temp_board[row],  # 가로
            [temp_board[i][col] for i in range(4)]  # 세로
        ]
        
        for line in lines_to_check:
            for dim in range(4):
                count = self.count_same_attribute_pieces(line, pieces, dim)
                if count == 4:  # 승리
                    score += 10.0
                elif count == 3:  # 거의 완성
                    score += 3.0
                elif count == 2:  # 발전 가능성
                    score += 0.5
        
        # 2. 대각선 평가 (해당하는 경우만)
        if row == col:  # 주대각선
            diag = [temp_board[i][i] for i in range(4)]
            for dim in range(4):
                count = self.count_same_attribute_pieces(diag, pieces, dim)
                if count == 4:
                    score += 10.0
                elif count == 3:
                    score += 3.0
                elif count == 2:
                    score += 0.5
        
        if row + col == 3:  # 부대각선
            diag = [temp_board[i][3-i] for i in range(4)]
            for dim in range(4):
                count = self.count_same_attribute_pieces(diag, pieces, dim)
                if count == 4:
                    score += 10.0
                elif count == 3:
                    score += 3.0
                elif count == 2:
                    score += 0.5
        
        # 3. 2x2 평가
        if row < 3 and col < 3:
            subgrid = [temp_board[row][col], temp_board[row][col+1],
                      temp_board[row+1][col], temp_board[row+1][col+1]]
            for dim in range(4):
                count = self.count_same_attribute_pieces(subgrid, pieces, dim)
                if count == 4:
                    score += 10.0
                elif count == 3:
                    score += 3.0
                elif count == 2:
                    score += 0.5
        
        return score
    
    def backpropagate(self, node: Node, result: float) -> None:
        """결과를 부모 노드들에게 전파"""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1 - result  # 결과 반전 (min-max)
    
    def get_best_move(self, root: Node, is_selection: bool = True):
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
        initial_state = (self.board, self.available_pieces)
        root = Node(initial_state, self.pieces, is_opponent_turn=True)
        return self.get_best_move(root, is_selection=True)
    
    def place_piece(self, selected_piece):
        """조각을 놓을 위치 선택"""
        initial_state = (self.board, [selected_piece])
        root = Node(initial_state, self.pieces, move=selected_piece, is_opponent_turn=False)
        return self.get_best_move(root, is_selection=False)