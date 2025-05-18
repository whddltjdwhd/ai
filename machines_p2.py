import numpy as np
from itertools import product
import time
import math
from copy import deepcopy
from typing import List, Tuple, Optional
import random

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
        self.untried_moves = self.get_possible_moves()
    
    def get_piece_index(self, piece, pieces):
        """조각의 인덱스 반환 (1-based)"""
        return pieces.index(piece) + 1
    
    def get_possible_moves(self):
        # self.move는 이 노드에 도달하기 위해 부모 노드에서 취한 행동입니다.

        # Case 1: 부모가 조각을 '선택'했고 (self.move가 piece_tuple), 이 조각이 현재 노드로 전달된 경우
        # 또는, 루트 노드인데 P2.place_piece()에서 호출되어 self.move가 배치할 조각인 경우.
        # 이 경우, 현재 노드에서는 해당 조각을 '배치'해야 합니다.
        # 조각의 특징: 튜플이고, 길이가 4이며, 모든 요소가 정수.
        if isinstance(self.move, tuple) and len(self.move) == 4 and all(isinstance(attr, int) for attr in self.move):
            return [(row, col) for row, col in product(range(4), range(4))
                   if self.board[row][col] == 0] # 가능한 모든 빈 위치 반환
        # Case 2: 부모가 조각을 '배치'했고 (self.move가 (piece, (r,c)) 형태),
        # 또는, 루트 노드인데 P2.select_piece()에서 호출되어 self.move가 None인 경우.
        # 이 경우, 현재 노드에서는 다음 조각을 '선택'해야 합니다.
        # 배치 행동의 특징: 튜플이고, 길이가 2이며, 첫번째 요소가 조각 튜플, 두번째 요소가 위치 튜플.
        else: # self.move가 (piece, (r,c)) 이거나 None인 경우 (루트에서 선택 시작)
            return list(self.available_pieces) # 사용 가능한 조각 목록 반환
    
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
    
    def get_uct_score(self, c_param=1.41, heuristic_weight=0.1):
        """UCT 점수 계산 (P2의 관점에서 평가 속성들을 보너스/패널티로 활용)"""
        if self.visits == 0:
            # 미방문 노드는 탐색을 장려하기 위해 UCT 점수로 float('inf')를 반환합니다. (튜닝 가능 지점)
            return float('inf') 
        if self.parent is None:
            return float('inf')
        
        # Exploitation Term (P2의 평균 승률)
        # self.wins는 항상 현재 MCTS 실행 주체(여기서는 P2)의 관점에서 기록된 승수라고 가정
        exploitation = self.wins / self.visits

        # Exploration Term
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)

        # Heuristic Bonus Term (P2의 관점에서)
        # win_potential: P2 승리 잠재력 (높을수록 좋음)
        # trap_potential: P2 트랩 잠재력 (높을수록 좋음) - evaluate_node에서 상황에 따라 설정됨
        # defense_value: P2 방어 가치 / P1에게 안전한 조각을 주는 가치 (높을수록 P2에게 좋음)
        # opponent_threat: P1(상대) 위협 수준 (높을수록 P2에게 안 좋음 -> 빼준다)
        heuristic_bonus = (self.win_potential + self.trap_potential + self.defense_value - self.opponent_threat) * heuristic_weight
        
        return exploitation + exploration + heuristic_bonus

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
        self.time_limit = 1.0  # 각 턴당 1.0초 사용 (여유 시간 확보)
    
    def get_piece_index(self, piece: Tuple, pieces_list: List[Tuple]) -> int:
        """조각의 인덱스 반환 (1-based)"""
        # self.pieces를 사용하지 않고, 인자로 받은 pieces_list를 사용하도록 명확히 함.
        # 이 pieces_list는 일반적으로 MCTS 노드나 다른 함수에서 전달되는 전체 조각 목록임.
        try:
            return pieces_list.index(piece) + 1
        except ValueError:
            # print(f"Error: Piece {piece} not found in provided pieces_list for get_piece_index.")
            # 디버깅을 위해 예외를 발생시키는 것이 좋을 수 있음.
            # 이 오류는 호출하는 쪽에서 piece와 pieces_list를 잘못 매칭했을 가능성을 시사함.
            raise ValueError(f"Piece {piece} not found in pieces_list during get_piece_index call.")

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
    
    def evaluate_p2_defense_value(self, r_placed: int, c_placed: int, piece_placed_by_p2: Tuple,
                                 board_before_p2_move: List[List[int]],
                                 all_game_pieces: List[Tuple],
                                 p1_potential_placements: List[Tuple]) -> float:
        """
        P2가 (r_placed, c_placed)에 piece_placed_by_p2를 놓은 행동의 방어적 가치를 P2 관점에서 평가.
        이 행동이 P1의 잠재적 위협(승리, 3목, 2x2의 3개 이상)을 얼마나 잘 차단했는지 평가.
        점수가 높을수록 P2의 방어가 성공적이었다는 의미입니다.
        P1의 `evaluate_defensive_value` 로직을 P2 관점으로 수정한 것입니다.
        `p1_potential_placements`는 P1이 `(r_placed, c_placed)`에 놓을 수 있었던 조각들입니다.
        """
        defense_score_for_p2 = 0.0
        
        # P2가 실제로 수를 둔 후의 보드는 이 함수 평가에는 직접 사용되지 않음.
        # 대신 P1이 가상으로 두는 상황과 P2가 둔 piece_placed_by_p2의 속성을 비교.

        for p1_hypothetical_piece in p1_potential_placements:
            # P1이 (r_placed, c_placed)에 p1_hypothetical_piece를 놓았다고 가정
            hypothetical_p1_board = deepcopy(board_before_p2_move)
            if hypothetical_p1_board[r_placed][c_placed] == 0: # 해당 칸이 비어있어야 P1이 놓을 수 있음
                hypothetical_p1_board[r_placed][c_placed] = self.get_piece_index(p1_hypothetical_piece, all_game_pieces)
            else:
                # 이 경우는 발생하면 안됨. board_before_p2_move는 P2가 두기 전이므로 해당 칸은 비어있어야 함.
                continue 

            p1_threat_blocked_by_this_piece = False

            # 1. P1의 잠재적 라인 위협 (승리 또는 3목) 및 P2의 차단 평가
            lines_indices_at_rc = []
            # 가로
            lines_indices_at_rc.append(hypothetical_p1_board[r_placed])
            # 세로
            lines_indices_at_rc.append([hypothetical_p1_board[i][c_placed] for i in range(4)])
            # 주대각선
            if r_placed == c_placed:
                lines_indices_at_rc.append([hypothetical_p1_board[i][i] for i in range(4)])
            # 부대각선
            if r_placed + c_placed == 3:
                lines_indices_at_rc.append([hypothetical_p1_board[i][3-i] for i in range(4)])

            for line_p1_formed_indices in lines_indices_at_rc:
                if p1_threat_blocked_by_this_piece: break
                if not line_p1_formed_indices: continue

                for dim in range(4):
                    # P1이 이 라인을 완성했는지 (4개) 또는 3개를 만들었는지 확인
                    # 실제 조각들을 가져와서 속성 비교
                    actual_pieces_in_hypo_line = []
                    for piece_idx in line_p1_formed_indices:
                        if piece_idx != 0:
                            actual_pieces_in_hypo_line.append(self.get_piece_from_index(piece_idx, all_game_pieces))
                    
                    if not actual_pieces_in_hypo_line: continue

                    # P1이 p1_hypothetical_piece를 놓음으로써 형성된 라인의 지배적인 속성 값
                    # 그리고 그 라인이 p1_hypothetical_piece의 속성을 따르는지 확인
                    p1_line_attr_val = p1_hypothetical_piece[dim]
                    is_p1_dominant_line = all(p[dim] == p1_line_attr_val for p in actual_pieces_in_hypo_line)

                    if is_p1_dominant_line:
                        if len(actual_pieces_in_hypo_line) == 4: # P1이 승리 라인 형성
                            if piece_placed_by_p2[dim] != p1_line_attr_val: # P2의 조각이 이 속성과 다르면 방어 성공
                                defense_score_for_p2 += 3.0
                                p1_threat_blocked_by_this_piece = True; break
                        elif len(actual_pieces_in_hypo_line) == 3 and (0 in line_p1_formed_indices or len(line_p1_formed_indices) != 4):
                            # P1이 3목 형성 (라인에 원래 빈칸이 있었거나, 라인 자체가 3칸짜리 대각선이 아닌 이상)
                            # P1의 원본은 0이 포함된 경우 3목으로 간주. 즉, 4칸 중 3개가 채워지고 모두 같은 속성.
                            # 좀 더 정확히는, 4칸짜리 슬롯에서 3개가 채워지고 그 3개가 p1_hypothetical_piece와 같은 속성.
                            # 위 is_p1_dominant_line && len == 3 조건으로 이미 충분.
                            if piece_placed_by_p2[dim] != p1_line_attr_val:
                                defense_score_for_p2 += 1.5
                                p1_threat_blocked_by_this_piece = True; break
                if p1_threat_blocked_by_this_piece: break
            if p1_threat_blocked_by_this_piece: continue # 다음 p1_hypothetical_piece로

            # 2. P1의 잠재적 2x2 위협 (3개 또는 4개 완성) 및 P2의 차단 평가
            # (r_placed, c_placed)를 포함하는 모든 2x2 격자 확인
            for r_offset in [-1, 0]:
                for c_offset in [-1, 0]:
                    if p1_threat_blocked_by_this_piece: break
                    r_start, c_start = r_placed + r_offset, c_placed + c_offset
                    if not (0 <= r_start <= 2 and 0 <= c_start <= 2): continue
                    # (r_placed, c_placed)가 현재 2x2 격자 (r_start, c_start)의 일부인지 확인
                    if not (r_start <= r_placed < r_start + 2 and c_start <= c_placed < c_start + 2):
                        continue

                    hypo_subgrid_indices = [
                        hypothetical_p1_board[r_start][c_start], hypothetical_p1_board[r_start][c_start+1],
                        hypothetical_p1_board[r_start+1][c_start], hypothetical_p1_board[r_start+1][c_start+1]
                    ]
                    
                    # P1의 로직은 2x2가 꽉 찼을 때만 고려 (if 0 in hypo_subgrid_indices: continue)
                    if 0 in hypo_subgrid_indices: continue
                    
                    hypo_subgrid_pieces = [self.get_piece_from_index(idx, all_game_pieces) for idx in hypo_subgrid_indices]

                    for dim_2x2 in range(4):
                        p1_2x2_attr_val = p1_hypothetical_piece[dim_2x2]
                        # 2x2 내의 조각들 중 p1_hypothetical_piece와 같은 속성을 가진 조각 수
                        count_p1_attr_in_2x2 = sum(1 for p in hypo_subgrid_pieces if p[dim_2x2] == p1_2x2_attr_val)
                        
                        if count_p1_attr_in_2x2 >= 3: # P1이 2x2에서 3개 또는 4개 완성 위협
                            if piece_placed_by_p2[dim_2x2] != p1_2x2_attr_val: # P2의 조각이 방해
                                defense_score_for_p2 += (2.0 if count_p1_attr_in_2x2 == 4 else 1.0)
                                p1_threat_blocked_by_this_piece = True; break
                    if p1_threat_blocked_by_this_piece: break
                if p1_threat_blocked_by_this_piece: break # 2x2 r_offset 루프 탈출
            # if p1_threat_blocked_by_this_piece: continue # 다음 p1_hypothetical_piece로 (이미 위에서 처리)

        return defense_score_for_p2

    def evaluate_node(self, node: Node) -> None:
        """
        노드의 전략적 가치 평가 (P2의 관점).
        이 노드는 'expand' 단계에서 방금 생성된 자식 노드입니다.
        평가 속성들은 P2에게 유리할수록 높은 값을 가집니다 (opponent_threat 제외).
        """
        action_that_led_to_node = node.move
        board_state_of_node = node.board
        all_game_pieces = node.pieces
        # 이 노드 상태에서 다음 플레이어가 선택/사용할 수 있는 조각들
        # (즉, node.is_opponent_turn이 True면 P1이 선택할 조각, False면 P2가 선택할 조각 - 현재 로직에서는 P2가 놓을 조각을 의미하기도 함)
        current_available_pieces_for_node = list(node.available_pieces)

        # 평가 속성 초기화 (P2 관점)
        node.win_potential = 0.0   # P2의 승리 잠재력
        node.trap_potential = 0.0  # P2의 트랩 잠재력
        node.defense_value = 0.0   # P2의 방어적 가치 / P1에게 안전한 조각을 주는 가치
        node.opponent_threat = 0.0 # P1(상대)의 위협 수준

        # node.is_opponent_turn이 False라는 것은, 이 노드에서 P2(나)가 행동할 차례라는 의미.
        # 즉, 이전 액션(action_that_led_to_node)은 P1(상대)이 수행한 것.
        if not node.is_opponent_turn:
            # Case 1: P1이 조각을 *선택*하여 P2에게 주었고 (`action_that_led_to_node`는 piece),
            # 이 `node`는 P2가 그 조각(piece_p1_gave)을 특정 위치에 *놓을* 차례인 상태.
            # (P1의 evaluate_node Case 4와 유사한 상황에서 P2의 관점)
            if not isinstance(action_that_led_to_node, tuple) and action_that_led_to_node is not None:
                piece_p1_gave_to_p2 = action_that_led_to_node # P1이 선택해서 P2에게 준 조각
                
                # P2의 승리 잠재력: P2가 이 조각을 놓아 이길 수 있는 최대 가능성.
                # (evaluate_p2_win_potential은 P2가 piece_p1_gave_to_p2를 여러 빈 칸에 놓았을 때의 최대 점수를 반환)
                node.win_potential = self.evaluate_p2_win_potential(piece_p1_gave_to_p2, board_state_of_node, all_game_pieces)
                
                # P1의 트랩 위협: P1이 준 조각이 P2에게 트랩인가?
                node.opponent_threat = self.evaluate_p1_trap_threat(piece_p1_gave_to_p2, board_state_of_node, all_game_pieces)
                
                # P2의 방어 가치: P1이 준 조각이 P1에게 얼마나 불리한(안전한) 조각인가?
                # evaluate_piece_safety_for_p1은 P1에게 유리한 정도이므로, -를 붙여 P2의 방어 가치로.
                node.defense_value = -self.evaluate_piece_safety_for_p1(piece_p1_gave_to_p2, board_state_of_node, all_game_pieces)
                
                node.trap_potential = 0.0 # P2가 놓는 상황이므로 P2의 트랩 설정은 아님.

            # Case 2: P1이 조각을 특정 위치에 *놓았고* (`action_that_led_to_node`는 (piece, (r,c))),
            # 이 `node`는 P2가 P1에게 줄 조각을 *선택할* 차례인 상태.
            # (P1의 evaluate_node Case 3와 유사한 상황에서 P2의 관점)
            elif isinstance(action_that_led_to_node, tuple) and len(action_that_led_to_node) == 2:
                piece_placed_by_p1, (r_p1, c_p1) = action_that_led_to_node
                
                # P1의 직접적인 위협: P1이 방금 놓은 수의 위협도.
                node.opponent_threat = self.evaluate_p1_threat(r_p1, c_p1, piece_placed_by_p1, board_state_of_node, all_game_pieces)
                
                # P2의 방어 가치: P2가 P1에게 줄 조각을 고르는 상황.
                # P1에게 가장 안전한(덜 위협적인) 조각을 주는 것이 P2의 방어적 행동.
                # current_available_pieces_for_node는 P2가 P1에게 줄 수 있는 조각 목록.
                # 그 중 P1에게 가장 안전한 조각을 주었을 때의 가치를 P2의 방어 가치로.
                if current_available_pieces_for_node:
                    min_p1_safety_score = float('inf')
                    for p_to_give_p1 in current_available_pieces_for_node:
                        safety = self.evaluate_piece_safety_for_p1(p_to_give_p1, board_state_of_node, all_game_pieces)
                        if safety < min_p1_safety_score:
                            min_p1_safety_score = safety
                    # min_p1_safety_score가 낮을수록 P2에게 유리. defense_value는 양수여야 하므로 - 붙임.
                    node.defense_value = -min_p1_safety_score 
                else:
                    node.defense_value = 0.0
                
                node.win_potential = 0.0  # P2가 선택 단계이므로 직접적인 승리 잠재력 없음.
                # P2의 트랩 잠재력: P2가 P1에게 준 조각이 P1에게 트랩을 형성하는가?
                # (P1이 이 조각을 어디에 놓든, 그 다음 P2가 이길 수 있는 상황)
                # 이 평가는 P1의 `evaluate_trap_potential`과 유사한 로직 필요. 우선 0.
                node.trap_potential = 0.0

        # node.is_opponent_turn이 True라는 것은, 이 노드에서 P1(상대)이 행동할 차례라는 의미.
        # 즉, 이전 액션(action_that_led_to_node)은 P2(나)가 수행한 것.
        else: 
            # Case 3: P2가 조각을 특정 위치에 *놓았고* (`action_that_led_to_node`는 (piece, (r,c))),
            # 이 `node`는 P1이 P2에게 줄 조각을 *선택할* 차례인 상태.
            # (P1의 evaluate_node Case 2와 유사한 상황에서 P2의 관점)
            if isinstance(action_that_led_to_node, tuple) and len(action_that_led_to_node) == 2:
                piece_placed_by_p2, (r_p2, c_p2) = action_that_led_to_node
                
                # P2의 승리 잠재력: P2가 방금 놓은 수의 라인 완성도/승리 가능성.
                # P1의 위협 평가 함수를 P2가 놓은 수에 대해 호출하여 P2의 잠재력으로 사용.
                node.win_potential = self.evaluate_p1_threat(r_p2, c_p2, piece_placed_by_p2, board_state_of_node, all_game_pieces)

                # P2의 방어 가치: P2가 놓은 수가 P1의 잠재적 위협을 얼마나 잘 막았는지.
                # node.parent가 있어야 board_before_p2_move를 가져올 수 있음. expand에서 생성된 child node이므로 parent 존재.
                # current_available_pieces_for_node는 P1이 P2에게 줄 조각 목록.
                if node.parent: # 루트 노드가 아닌 경우
                    board_before_p2_action = node.parent.board
                    # P1이 다음 턴에 P2에게 줄 수 있는 조각 (P1이 지금 고를 조각) = current_available_pieces_for_node
                    # 이 조각들은 P1이 (r_p2, c_p2)에 놓았을 경우를 가정하는 p1_potential_placements가 됨.
                    node.defense_value = self.evaluate_p2_defense_value(r_p2, c_p2, piece_placed_by_p2, 
                                                                      board_before_p2_action, all_game_pieces, 
                                                                      current_available_pieces_for_node)
                else: # 루트 노드 등 parent가 없는 예외적인 경우
                    node.defense_value = 0.0
                
                # P1의 위협: P1이 이제 조각을 선택할 차례. P1이 P2에게 줄 조각이 P2에게 얼마나 위협적일까?
                # P1이 선택 가능한 조각들(current_available_pieces_for_node) 중 P2에게 가장 위협적인(P1에게 트랩을 주거나, P2의 승리 잠재력을 높이는) 조각의 위험도.
                # P1은 P2에게 트랩이 되는 조각을 주거나, P2가 놓았을 때 P2의 승리 가능성이 낮은 조각을 줄 것임.
                # 여기서는 P1이 P2에게 줄 조각이 P2에게 얼마나 위협적인지(P1이 트랩을 놓는지)를 평가.
                if current_available_pieces_for_node:
                    max_p1_trap_threat_to_p2 = float('-inf')
                    for p1_choice_for_p2 in current_available_pieces_for_node:
                        threat = self.evaluate_p1_trap_threat(p1_choice_for_p2, board_state_of_node, all_game_pieces)
                        if threat > max_p1_trap_threat_to_p2:
                            max_p1_trap_threat_to_p2 = threat
                    node.opponent_threat = max_p1_trap_threat_to_p2
                else:
                    node.opponent_threat = 0.0
                
                node.trap_potential = 0.0 # P2가 방금 두었고 P1이 선택할 차례이므로 P2의 트랩 잠재력은 0
            
            # Case 4: P2가 조각을 *선택*하여 P1에게 주었고 (`action_that_led_to_node`는 piece),
            # 이 `node`는 P1이 그 조각(piece_p2_selected)을 특정 위치에 *놓을* 차례인 상태.
            # (P1의 evaluate_node Case 1와 유사한 상황에서 P2의 관점)
            elif not isinstance(action_that_led_to_node, tuple) and action_that_led_to_node is not None:
                piece_p2_selected_for_p1 = action_that_led_to_node # P2가 선택해서 P1에게 준 조각

                # P1의 위협: P1이 P2가 준 조각으로 얼마나 이길 수 있나?
                # (evaluate_p2_win_potential 함수는 특정 조각을 놓았을 때의 승리 잠재력을 평가하므로, P1의 승리 잠재력으로 사용 가능)
                node.opponent_threat = self.evaluate_p2_win_potential(piece_p2_selected_for_p1, board_state_of_node, all_game_pieces)

                # P2의 방어 가치: P2가 P1에게 준 조각이 P1에게 얼마나 불리한(안전한) 조각이었나?
                node.defense_value = -self.evaluate_piece_safety_for_p1(piece_p2_selected_for_p1, board_state_of_node, all_game_pieces)
                
                # P2의 트랩 잠재력: P2가 P1에게 준 조각이 P1에게 트랩을 형성하는가?
                # (P1이 이 조각을 어디에 놓든, 그 다음 P2가 이길 수 있는 상황)
                # 이 평가는 P1의 `evaluate_trap_potential`과 유사한 로직 필요. 우선 0.
                node.trap_potential = 0.0
                node.win_potential = 0.0 # P1이 놓을 차례이므로 P2의 직접적인 승리 잠재력 0.

    def select(self, node):
        """UCT 값이 가장 높은 자식 노드 선택"""
        while not node.is_terminal(self.check_win):
            if not node.is_fully_expanded():
                return node
            node = max(node.children, key=lambda n: n.get_uct_score())
        return node
    
    def expand(self, node: Node) -> Node:
        """주어진 노드에서 아직 시도하지 않은 수 중 하나를 선택하여 자식 노드 생성"""
        # P2의 Node 클래스는 untried_moves를 Node 생성 시 _get_possible_moves()로 초기화.
        # 그리고 Node에는 untried_moves를 직접 수정하는 로직은 없었음.
        # 따라서 expand 함수는 node.untried_moves를 읽기만 하고, 그 중 하나를 선택.
        # 선택된 move를 Node에서 제거하는 로직은 expand 내에 있거나, 혹은 Node에 추가되어야 함.
        # 이전 P2 expand 로직에서는 possible_moves와 children을 비교하여 untried_moves를 동적으로 계산.
        # 지금은 node.untried_moves 속성이 이미 존재하고, 올바르게 관리된다고 가정.

        if not node.untried_moves: # 시도하지 않은 수가 없으면
            return node # 현재 노드를 반환 (새로운 자식 없음)

        # 시도하지 않은 수 목록에서 무작위로 하나의 수를 선택
        selected_move = random.choice(node.untried_moves)

        # P2의 이전 expand 로직에서는 여기서 try-except로 make_move 및 Node 생성을 감쌌음.
        try:
            # 새로운 게임 상태 계산
            next_state = self.make_move((deepcopy(node.board), list(node.available_pieces)), selected_move, self.pieces)

            # 새로운 노드의 is_opponent_turn 계산 (항상 부모와 반대)
            child_is_opponent_turn = not node.is_opponent_turn

            # 새로운 노드 생성
            new_node = Node(next_state, self.pieces, parent=node, move=selected_move, is_opponent_turn=child_is_opponent_turn)

            # 자식 노드 평가 및 추가
            self.evaluate_node(new_node) # P2의 평가 함수
            node.children.append(new_node)
            return new_node # 새로 생성된 자식 노드 반환
        except Exception as e:
            # print(f"Error during expansion with move {selected_move}: {e}")
            # 오류 발생 시, 해당 move는 문제가 있는 것으로 간주하고 현재 node를 반환하여
            # MCTS 루프가 계속 다른 선택을 시도하도록 할 수 있음.
            # 혹은 오류를 일으킨 move를 node.untried_moves에서 제거하는 로직도 고려 가능 (Node 수정 필요)
            return node # 오류 시 현재 노드 반환
    
    def simulate(self, node: Node) -> float:
        """
        MCTS Simulation 단계: 현재 노드부터 게임 종료까지 플레이아웃 진행 (휴리스틱 기반)
        이 함수는 Quarto 게임의 [선택 -> 배치] 턴 순서를 따르며,
        각 플레이어의 휴리스틱 플레이아웃 함수를 호출합니다.
        :param node: 시뮬레이션을 시작할 노드
        :return: 시뮬레이션 결과 점수 (P2 관점: 1.0: 승리, 0.5: 무승부, 0.0: 패배)
        """
        # --- 시뮬레이션 상태 초기화 --- 
        current_board = deepcopy(node.board)
        current_available_pieces = list(node.available_pieces)
        all_pieces = node.pieces # 전체 조각 정보

        # is_current_turn_p1: 현재 턴이 P1의 턴인가?
        is_current_turn_p1 = node.is_opponent_turn

        # action_is_selection: 현재 턴이 조각 '선택' 단계인가?
        piece_to_place_next = None # 현재 배치해야 할 조각
        if isinstance(node.move, tuple) and len(node.move) == 2: # 이전 행동이 배치였음 -> 이번엔 선택
            action_is_selection = True
        else: # 이전 행동이 선택(piece)이었거나 루트 노드
            action_is_selection = False # 이번엔 배치
            if node.move is not None and not isinstance(node.move, tuple):
                piece_to_place_next = node.move # 이전 액션에서 선택된 조각
            elif node.parent is None and node.move is not None: # 루트 노드인데 place_piece로 시작한 경우
                 piece_to_place_next = node.move
        
        # --- 시뮬레이션 루프 --- 
        while True: # 게임 종료 또는 진행 불가 시 내부에서 break
            
            # 게임 종료 조건 확인 (배치 직후에만 확인해도 되지만, 매번 확인해도 무방)
            if self.check_win(current_board, all_pieces):
                 break 

            # 진행 불가 조건 확인
            if action_is_selection and not current_available_pieces:
                # 선택 단계인데 선택할 조각이 없으면 무승부
                break
            if not action_is_selection and piece_to_place_next is None:
                # 배치 단계인데 배치할 조각 정보가 없으면 오류 또는 무승부
                 if not any(val == 0 for row_val in current_board for val in row_val):
                     # 보드가 꽉 찼으면 무승부
                     break
                 else:
                     # print("Simulate Error: piece_to_place_next is None in placement phase.")
                     # 오류 상황, 무승부 처리 또는 예외 발생 필요
                     break 
            if not action_is_selection and not any(val == 0 for row_val in current_board for val in row_val):
                 # 배치 단계인데 놓을 칸이 없으면 무승부
                 break
            
            # --- 현재 턴의 액션 결정 --- 
            current_move = None
            if is_current_turn_p1:
                # P1 턴
                if action_is_selection: # P1이 P2에게 줄 조각 선택
                    selected_piece = self.p1_select_piece_playout(current_board, current_available_pieces, all_pieces)
                    if selected_piece is None: break # 선택 불가
                    current_move = selected_piece
                    piece_to_place_next = selected_piece # 다음 P2 배치 턴에 사용될 조각
                else: # P1이 P2가 준 조각(piece_to_place_next) 배치
                    if piece_to_place_next is None: break # 오류
                    position = self.opponent_playout(current_board, piece_to_place_next, all_pieces)
                    if position is None: break # 배치 불가
                    current_move = (piece_to_place_next, position)
                    piece_to_place_next = None # 조각 사용 완료
            else:
                # P2 턴
                if action_is_selection: # P2가 P1에게 줄 조각 선택
                    selected_piece = self.heuristic_playout(current_board, current_available_pieces, all_pieces)
                    if selected_piece is None: break # 선택 불가
                    current_move = selected_piece
                    piece_to_place_next = selected_piece # 다음 P1 배치 턴에 사용될 조각
                else: # P2가 P1이 준 조각(piece_to_place_next) 배치
                    if piece_to_place_next is None: break # 오류
                    position = self.place_piece_playout(current_board, piece_to_place_next, all_pieces)
                    if position is None: break # 배치 불가
                    current_move = (piece_to_place_next, position)
                    piece_to_place_next = None # 조각 사용 완료
            
            if current_move is None: # 액션 결정 실패
                break

            # --- 상태 업데이트 --- 
            current_state = (current_board, tuple(current_available_pieces))
            next_state = self.make_move(current_state, current_move, all_pieces)
            current_board = next_state[0]
            current_available_pieces = list(next_state[1])

            # --- 턴 및 단계 전환 --- 
            action_is_selection = not action_is_selection # 선택 <-> 배치 전환
            if action_is_selection: # 배치가 끝났으면 (다음은 선택), 턴 전환
                is_current_turn_p1 = not is_current_turn_p1

        # --- 시뮬레이션 결과 반환 --- 
        if self.check_win(current_board, all_pieces):
            # is_current_turn_p1은 '다음' 행동할 플레이어
            # 승리 보드를 만든 플레이어는 직전 턴 플레이어 = not is_current_turn_p1
            winner_is_p1 = not is_current_turn_p1
            return 0.0 if winner_is_p1 else 1.0 # P2 관점 결과 반환
        else:
            # 무승부 (진행 불가 또는 보드 꽉 참)
            return 0.5
    
    def heuristic_playout(self, board: List[List[int]], available_pieces: List[Tuple], 
                         pieces: List[Tuple]) -> Tuple:
        """
        AI (P2)가 P1에게 줄 조각을 선택하는 휴리스틱 기반 시뮬레이션.
        P1에게 가장 안전한(덜 위협적인) 조각을 선택하는 것을 목표로 합니다.
        :return: 선택된 조각 (P1에게 줄 조각)
        """
        best_piece_to_give_p1 = None
        # P2는 P1에게 가장 안전한(P1에게 점수가 낮은) 조각을 주려고 하므로, 초기 best_score는 매우 높아야 함.
        lowest_p1_advantage_score = float('inf') 

        if not available_pieces: # 줄 수 있는 조각이 없으면 랜덤 선택 방지 (실제로는 발생 안해야함)
             # 이 상황은 게임 로직상 거의 발생하지 않지만, 방어 코드
            return pieces[0] if pieces else (0,0,0,0) # 임의의 기본값

        for piece in available_pieces:
            # P1이 이 조각을 받았을 때 P1에게 얼마나 유리한지 평가
            # evaluate_piece_safety_for_p1은 높을수록 P1에게 유리
            p1_advantage = self.evaluate_piece_safety_for_p1(piece, board, pieces)
            
            # 추가적으로, P1이 이 조각으로 즉시 트랩을 만들 수 있는지도 고려 (낮을수록 좋음)
            # evaluate_p1_trap_threat는 P1이 piece를 P2에게 주었을 때의 위협이므로 여기서는 직접 사용 부적절.
            # 대신, 만약 P1이 이 piece를 받아서 특정 위치에 놓았을 때, 다음 P2가 줄 조각으로 P1이 이기는 상황이 되는가?
            # 이 부분은 더 복잡한 평가가 필요하므로, 우선 piece_safety_for_p1에 집중.

            current_score_for_p1 = p1_advantage
            # TODO: 여기에 P1에게 불리함을 더하는 다른 휴리스틱 고려 가능 (예: P1의 선택지를 줄이는 조각?)

            if current_score_for_p1 < lowest_p1_advantage_score:
                lowest_p1_advantage_score = current_score_for_p1
                best_piece_to_give_p1 = piece
        
        # 만약 모든 조각이 동일한 매우 높은 점수를 갖거나 문제가 발생하면 랜덤 선택
        return best_piece_to_give_p1 if best_piece_to_give_p1 else available_pieces[np.random.randint(len(available_pieces))]
    
    def opponent_playout(self, board: List[List[int]], piece_from_p2: Tuple, 
                        pieces: List[Tuple]) -> Optional[Tuple[int, int]]:
        """
        상대방(P1)이 P2로부터 받은 `piece_from_p2`를 보드에 놓을 위치를 결정하는 휴리스틱.
        P1의 공격적 성향(즉시 승리, 트랩, 라인 완성)을 시뮬레이션합니다.
        :return: 선택된 위치 (row, col) 또는 None
        """
        available_locs = [(r, c) for r, c in product(range(4), range(4)) 
                         if board[r][c] == 0]
        if not available_locs:
            return None
            
        best_p1_move = None
        best_p1_score = float('-inf') # P1에게 가장 좋은 수를 찾음
        
        for r, c in available_locs:
            p1_score_at_loc = 0
            
            # 1. P1이 즉시 승리 가능한 위치 선호
            temp_board_p1_wins = deepcopy(board)
            temp_board_p1_wins[r][c] = self.get_piece_index(piece_from_p2, pieces)
            if self.check_win(temp_board_p1_wins, pieces):
                return (r, c) # 즉시 승리 가능하면 바로 선택
            
            # 2. P1이 트랩을 만들 수 있는 위치 선호 (P1 관점의 트랩)
            # evaluate_trap_potential_at은 P1의 관점에서 점수를 반환 (가정)
            p1_score_at_loc += self.evaluate_trap_potential_at(r, c, piece_from_p2, board, pieces) * 2.0
            
            # 3. 중앙에 가까운 위치 선호 (전략적 중요성)
            center_dist = abs(r - 1.5) + abs(c - 1.5)
            p1_score_at_loc += (4 - center_dist) * 0.5
            
            # 4. P1의 라인 완성 가능성이 높은 위치 선호
            # evaluate_p1_threat는 (r,c)에 piece를 놓았을 때의 라인 완성도(위협도)를 P1 관점에서 평가
            p1_score_at_loc += self.evaluate_p2_line_completion(r, c, piece_from_p2, board, pieces) * 1.0
            
            if p1_score_at_loc > best_p1_score:
                best_p1_score = p1_score_at_loc
                best_p1_move = (r, c)
        
        return best_p1_move if best_p1_move else available_locs[np.random.randint(len(available_locs))]
    
    def evaluate_simulation_result(self, board: List[List[int]], pieces: List[Tuple],
                                 is_opponent_turn: bool) -> float:
        """
        시뮬레이션 결과 평가
        :return: AI(P2) 관점에서의 점수 (1.0: 승리, 0.5: 무승부, 0.0: 패배)
        """
        if self.check_win(board, pieces):
            return 0.0 if is_opponent_turn else 1.0  # 마지막 수를 둔 플레이어의 승리
        return 0.5  # 무승부
    
    def _count_same_attribute_pieces(self, line: List[int], pieces: List[Tuple], 
                                   target_dim: int) -> int:
        """
        주어진 라인에서 특정 속성(target_dim)이 같은 값을 가지는 조각의 수를 센다.
        빈 칸(0)은 세지 않는다.
        """
        count = 0
        first_piece_attribute = None
        
        line_pieces_attributes = []
        for piece_idx in line:
            if piece_idx == 0: # 빈 칸
                continue
            piece = pieces[piece_idx-1] # 1-based index to 0-based
            line_pieces_attributes.append(piece[target_dim])

        if not line_pieces_attributes:
            return 0

        # 첫 번째 조각의 속성을 기준으로 삼는다.
        # 만약 라인에 모든 조각이 특정 속성에 대해 동일한 값을 가지는지 확인하려면
        # 모든 조각의 해당 속성 값이 첫 번째 조각의 속성 값과 같은지 확인하면 된다.
        # 여기서는 단순히 '같은 속성을 공유하는 조각의 최대 그룹 크기'를 찾는 것이 아니라,
        # 특정 속성값이 같은 조각의 '개수'를 세는 것이므로,
        # 라인 내 모든 조각이 해당 차원에서 모두 같은 값을 가져야 의미가 있다.
        # 따라서, check_line_attributes 와 유사하게 첫번째 값을 기준으로 동일한지 본다.
        
        # P1의 원본 로직은 한 라인에서 특정 속성이 같은 *모든* 조각의 수를 세는 것이 아니라,
        # 해당 속성에 대해 '모든 조각이 동일한 값'을 가지는 경우, 그 라인의 조각 수를 반환하는 것에 가까웠다.
        # 여기서는 해당 차원에서 특정 값을 공유하는 조각의 수를 세도록 수정한다. (예: 높이가 '높음'인 조각이 몇 개인지)
        # 좀 더 정확히는, 한 줄에서 특정 차원의 속성 값이 같은 조각들의 최대 개수를 센다.
        
        # 다시 생각해보니, P1의 evaluate_winning_potential 에서의 count는 
        # "해당 라인에서, 특정 차원의 속성값이 모두 동일한 조각들의 개수"를 의미했다.
        # 예를 들어 [A, A, A, B] 이고 A의 특정 속성이 같다면 3을 반환하는게 아니라,
        # [A, A, A, A] 처럼 모두 같을 때만 4를 반환하고, 그렇지 않으면 의미 없는 값(0)을 반환해야 한다.
        # P1의 check_line_attributes가 그런 역할을 한다.
        # count_same_attribute_pieces는 check_line_attributes를 사용하는 것이 아니라,
        # 해당 라인에 있는 조각들 중, 특정 속성(dim)에 대해 *값이 같은* 조각이 몇개인지 센다.
        # 예를 들어, 4칸 중 3칸이 차있고, 그 3개의 조각이 모두 '높음' 속성을 공유하면 3을 반환.
        
        # P1의 `count_same_attribute_pieces`는 사실상 `check_line_attributes`와 유사하게,
        # 라인 내 모든 조각이 특정 차원에서 동일한 속성 값을 가질 때만 유효한 카운트(라인 길이)를 반환하고,
        # 그렇지 않으면 0이나 작은 수를 반환하는 형태로 동작했을 가능성이 높다.
        # 여기서는 좀 더 명확하게, "특정 속성 값이 같은 조각의 수"를 계산한다.
        # 하지만 P1의 코드를 그대로 가져오면서 점수 로직이 count == 4, count == 3 등에 의존하므로,
        # P1과 동일한 방식으로 작동하도록 유지한다. 즉, 해당 라인의 모든 조각이 특정 속성에서 동일한 값을 가질 때만
        # 그 개수(라인의 길이)를 반환하고, 하나라도 다르면 0을 반환하는 식으로 가정한다.

        # P1 코드의 의도를 최대한 살려서, 라인 내의 조각들이 해당 차원에서 모두 같은 속성값을 가지는지 확인하고,
        # 그렇다면 그 조각들의 수를 반환. 아니면 0 반환.
        # 단, 빈 칸은 제외하고 실제 조각들만 대상으로 한다.
        
        actual_pieces_in_line = []
        for piece_idx in line:
            if piece_idx != 0:
                actual_pieces_in_line.append(pieces[piece_idx-1])
        
        if not actual_pieces_in_line:
            return 0

        first_val = actual_pieces_in_line[0][target_dim]
        if all(p[target_dim] == first_val for p in actual_pieces_in_line):
            return len(actual_pieces_in_line)
        return 0

    def evaluate_p2_win_potential(self, piece: Tuple, board: List[List[int]], 
                                 pieces: List[Tuple]) -> float:
        """
        P2가 특정 `piece`를 놓았을 때 P2 자신의 승리 잠재력 평가.
        이 조각을 놓을 수 있는 각 위치에서 새로운 3목/4목 또는 2x2 완성이 만들어질 가능성을 P2 관점에서 평가합니다.
        높은 점수는 P2에게 유리한 상황을 의미합니다.
        (P1의 `evaluate_winning_potential` 로직 기반)
        """
        score = 0.0
        
        # 1. 각 빈칸에 이 조각을 놓아보고 평가
        for r_idx in range(4):
            for c_idx in range(4):
                if board[r_idx][c_idx] != 0:  # 이미 채워진 칸은 건너뛰기
                    continue
                    
                # 임시로 조각을 놓아보기
                temp_board = deepcopy(board)
                # P2 클래스에는 get_piece_index가 있으므로 사용
                temp_board[r_idx][c_idx] = self.get_piece_index(piece, pieces) 
                
                # 2. 이 위치를 지나는 모든 라인 평가
                lines_to_check = []
                # 가로 라인
                lines_to_check.append(temp_board[r_idx])
                # 세로 라인
                lines_to_check.append([temp_board[i][c_idx] for i in range(4)])
                # 주대각선 (해당되는 경우)
                if r_idx == c_idx:
                    lines_to_check.append([temp_board[i][i] for i in range(4)])
                # 부대각선 (해당되는 경우)
                if r_idx + c_idx == 3:
                    lines_to_check.append([temp_board[i][3-i] for i in range(4)])
                
                # 3. 각 라인에서 같은 속성을 가진 조각 수 평가 (P2 관점)
                for line_indices in lines_to_check:
                    if not line_indices: 
                        continue
                    for dim in range(4): # 4가지 속성 차원
                        # _count_same_attribute_pieces는 해당 라인의 모든 조각이 특정 속성에서 동일한 값을 가질 때 그 개수를 반환
                        count = self._count_same_attribute_pieces(line_indices, pieces, dim)
                        if count == 4:  # P2 승리
                            score += 10.0
                        elif count == 3:  # P2의 잠재적 승리 (3개 완성)
                            score += 3.0
                        elif count == 2:  # P2의 발전 가능성 (2개 완성)
                            score += 0.5
                
                # 4. 2x2 부분 격자 평가 (P2 관점)
                # (r_idx, c_idx)가 좌상단 모서리가 되는 2x2 격자들을 검사
                # 이 로직은 piece가 (r_idx, c_idx)에 놓였을 때, 이 piece를 포함하는 모든 2x2를 검사해야 함.
                # (r,c)가 (r_idx, c_idx)인 2x2는 1개 (만약 r_idx,c_idx < 3)
                # (r,c)가 (r_idx-1, c_idx)인 2x2는 1개 (만약 r_idx > 0, c_idx < 3)
                # (r,c)가 (r_idx, c_idx-1)인 2x2는 1개 (만약 r_idx < 3, c_idx > 0)
                # (r,c)가 (r_idx-1, c_idx-1)인 2x2는 1개 (만약 r_idx > 0, c_idx > 0)
                
                for dr in [-1, 0]: # 2x2의 좌상단 r좌표를 기준으로 현재 놓은 위치와의 상대적 위치
                    for dc in [-1, 0]: # 2x2의 좌상단 c좌표를 기준으로 현재 놓은 위치와의 상대적 위치
                        r_start, c_start = r_idx + dr, c_idx + dc
                        if 0 <= r_start <= 2 and 0 <= c_start <= 2: # 유효한 2x2 좌상단인가?
                            # 현재 piece가 이 2x2 격자 내에 포함되는지 확인
                            if not (r_start <= r_idx < r_start + 2 and c_start <= c_idx < c_start + 2):
                                continue

                            subgrid_indices = [temp_board[r_start][c_start], temp_board[r_start][c_start+1],
                                             temp_board[r_start+1][c_start], temp_board[r_start+1][c_start+1]]
                            
                            if 0 in subgrid_indices: # 아직 2x2가 다 채워지지 않음 (piece를 놓았음에도)
                                # 사실 이 piece를 놓음으로써 2x2가 완성될 수도 있으므로, 0의 개수를 세는게 더 정확.
                                # 만약 piece를 놓은 후에도 0이 있다면, 그 2x2는 아직 미완성.
                                # P1의 로직은 piece를 놓은 *후*의 subgrid를 평가.
                                pass # 이미 위에서 temp_board를 사용하므로 0이 있으면 완성 안됨.

                            for dim in range(4):
                                count = self._count_same_attribute_pieces(subgrid_indices, pieces, dim)
                                if count == 4:  # P2의 2x2 승리
                                    score += 10.0 
                                elif count == 3:  # P2의 잠재적 2x2 (3개 완성)
                                    # 2x2에서 3개가 특정 속성을 공유하면, 나머지 하나만 맞으면 승리.
                                    score += 3.0 
                                    # P1의 코드는 2x2에서 3개가 모이면 바로 3점.
        
        return score

    def evaluate_winning_potential(self, piece: Tuple, pieces: List[Tuple]) -> float:
        """조각의 승리 잠재력 평가"""
        # TODO: 구현 예정 (이 함수는 evaluate_p2_win_potential로 대체되었으므로 삭제 또는 주석 처리 예정)
        return 0.0
    
    def evaluate_piece_safety_for_p1(self, piece_to_give: Tuple, board: List[List[int]], 
                                  pieces: List[Tuple]) -> float:
        """
        P2가 P1에게 줄 `piece_to_give` 조각이 P1에게 얼마나 불리한(안전한) 조각인지 평가합니다.
        P1이 이 조각을 빈 칸에 놓았을 때, P1에게 유리한 4개/3개/2개 완성이 얼마나 많이 발생하는지를 체크하고 점수를 부여합니다.
        이 점수는 P1에게 해당 조각이 얼마나 매력적인지를 나타냅니다.
        P2 관점에서는 이 점수가 낮을수록 P2가 주기에 좋은(안전한) 조각입니다.
        (P1의 `evaluate_winning_potential` 로직과 유사)
        """
        p1_advantage_score = 0.0
        
        # 1. 각 빈칸에 P1이 piece_to_give를 놓는다고 가정하고 평가
        for r_idx in range(4):
            for c_idx in range(4):
                if board[r_idx][c_idx] != 0:  # 이미 채워진 칸은 건너뛰기
                    continue
                    
                # 임시로 P1이 조각을 놓아보기
                temp_board = deepcopy(board)
                temp_board[r_idx][c_idx] = self.get_piece_index(piece_to_give, pieces)
                
                # 2. 이 위치를 지나는 모든 라인 평가 (P1 관점의 이득)
                lines_to_check = []
                lines_to_check.append(temp_board[r_idx]) # 가로
                lines_to_check.append([temp_board[i][c_idx] for i in range(4)]) # 세로
                if r_idx == c_idx: lines_to_check.append([temp_board[i][i] for i in range(4)]) # 주대각선
                if r_idx + c_idx == 3: lines_to_check.append([temp_board[i][3-i] for i in range(4)]) # 부대각선
                
                for line_indices in lines_to_check:
                    if len(line_indices) == 0: continue
                    for dim in range(4):
                        count = self._count_same_attribute_pieces(line_indices, pieces, dim)
                        if count == 4:  # P1이 이 조각으로 즉시 승리
                            p1_advantage_score += 10.0
                        elif count == 3:  # P1이 3목 형성
                            p1_advantage_score += 3.0
                        elif count == 2:  # P1이 2목 형성
                            p1_advantage_score += 0.5
                
                # 3. 2x2 부분 격자 평가 (P1 관점의 이득)
                # piece_to_give가 (r_idx, c_idx)에 놓였을 때, 이 piece를 포함하는 모든 2x2를 검사
                for dr in [-1, 0]:
                    for dc in [-1, 0]:
                        r_start, c_start = r_idx + dr, c_idx + dc
                        if 0 <= r_start <= 2 and 0 <= c_start <= 2:
                            if not (r_start <= r_idx < r_start + 2 and c_start <= c_idx < c_start + 2):
                                continue
                            subgrid_indices = [temp_board[r_start][c_start], temp_board[r_start][c_start+1],
                                             temp_board[r_start+1][c_start], temp_board[r_start+1][c_start+1]]
                            for dim in range(4):
                                count = self._count_same_attribute_pieces(subgrid_indices, pieces, dim)
                                if count == 4:  # P1이 2x2 승리
                                    p1_advantage_score += 10.0 
                                elif count == 3:  # P1이 2x2에서 3개 완성
                                    p1_advantage_score += 3.0
                                # elif count == 2: # 2x2에서 2개 완성은 P1의 winning_potential에서는 0.5점이었으나, 여기서는 일단 제외 (필요시 추가)
                                # P1의 evaluate_line_completion 및 evaluate_winning_potential 에서는 2x2의 2개 완성도 점수 부여.
                                # 일관성을 위해 여기도 포함.
                                elif count == 2:
                                     p1_advantage_score += 0.5
        
        return p1_advantage_score # 높은 점수 = P1에게 유리한 조각

    def evaluate_p1_trap_threat(self, piece_p1_gave: Tuple, board: List[List[int]], 
                              all_pieces_info: List[Tuple]) -> float:
        """
        P1이 P2에게 준 `piece_p1_gave` 조각이 P2에게 트랩을 형성하는 조각인지 평가합니다.
        P2가 `piece_p1_gave`를 놓을 수 있는 모든 빈 칸에 대해 시뮬레이션하고, 
        각 시뮬레이션 결과 보드에서 P1이 'P1이 배치할 수 있는 남은 조각'들로 바로 승리할 수 있는지 확인합니다.
        점수가 높을수록 P2에게 위험한 상황을 의미합니다.
        """
        trap_score = 0.0
        
        empty_cells_for_p2 = []
        for r in range(4):
            for c in range(4):
                if board[r][c] == 0:
                    empty_cells_for_p2.append((r, c))

        if not empty_cells_for_p2:
            return 0.0 # P2가 놓을 곳이 없으면 트랩 평가는 무의미

        # 현재 보드에 놓인 조각들 (튜플 형태)
        current_board_piece_indices = {idx for row_val in board for idx in row_val if idx != 0}
        current_board_pieces_tuples = {all_pieces_info[idx-1] for idx in current_board_piece_indices}
        
        # P1이 다음 턴에 배치하여 승리할 수 있는 후보 조각들
        # (전체 조각 중, 현재 보드에 없고, P1이 방금 P2에게 준 조각도 아닌 것들)
        p1_candidate_placement_pieces = [
            p for p in all_pieces_info 
            if p not in current_board_pieces_tuples and p != piece_p1_gave
        ]

        trapped_p2_placements_count = 0
        total_p2_placement_options = len(empty_cells_for_p2)

        for r_p2, c_p2 in empty_cells_for_p2:
            # P2가 piece_p1_gave를 (r_p2, c_p2)에 놓았다고 가정
            board_after_p2_places = deepcopy(board)
            board_after_p2_places[r_p2][c_p2] = self.get_piece_index(piece_p1_gave, all_pieces_info)
            
            p1_can_win_this_scenario = False
            # P1이 자신의 다음 턴에 이길 수 있는지 확인
            empty_cells_for_p1_to_place = []
            for r_p1 in range(4):
                for c_p1 in range(4):
                    if board_after_p2_places[r_p1][c_p1] == 0:
                        empty_cells_for_p1_to_place.append((r_p1, c_p1))
            
            if not empty_cells_for_p1_to_place: # P1이 놓을 곳이 없으면 P1은 못 이김
                continue

            for p1_piece_to_place in p1_candidate_placement_pieces:
                if p1_can_win_this_scenario: break
                for r_p1_place, c_p1_place in empty_cells_for_p1_to_place:
                    final_board_state = deepcopy(board_after_p2_places)
                    final_board_state[r_p1_place][c_p1_place] = self.get_piece_index(p1_piece_to_place, all_pieces_info)
                    if self.check_win(final_board_state, all_pieces_info):
                        p1_can_win_this_scenario = True
                        break # P1 승리, 다음 P2 배치 시나리오로
            
            if p1_can_win_this_scenario:
                trapped_p2_placements_count += 1

        if total_p2_placement_options == 0: # 이 경우는 위에서 처리했지만 안전장치
            return 0.0
        
        # P2의 모든 가능한 수가 P1의 승리로 이어진다면 최대 점수 (10.0)
        # P2의 어떤 수도 P1의 승리로 이어지지 않는다면 0점
        trap_score = 10.0 * (trapped_p2_placements_count / total_p2_placement_options)
        
        return trap_score

    def evaluate_blocking_potential(self, piece: Tuple, pieces: List[Tuple]) -> float:
        """조각의 방어 잠재력 평가"""
        # TODO: 구현 예정 (이 함수는 evaluate_piece_safety_for_p1로 대체되었으므로 삭제 또는 주석 처리 예정)
        return 0.0
    
    def evaluate_safety(self, piece: Tuple, board: List[List[int]], 
                       pieces: List[Tuple]) -> float:
        """조각의 안전성 평가"""
        # TODO: 구현 예정
        return 0.0
    
    def evaluate_trap_potential_at(self, row: int, col: int, piece: Tuple, 
                                 board: List[List[int]], pieces: List[Tuple]) -> float:
        """특정 위치의 트랩 생성 잠재력 평가"""
        # 이 함수는 현재 opponent_playout (P1의 배치 시뮬레이션)에서 호출될 수 있으나,
        # P1의 원래 휴리스틱에는 이와 직접 대응되는 함수가 없었습니다.
        # P2의 방어적 관점에서 P1이 만들 수 있는 트랩을 평가하기 위해 정의될 수 있으나,
        # 현재는 기본값 0.0을 반환합니다.
        # 향후 P1의 공격적 트랩 생성 로직(예: evaluate_trap_potential)을 참고하여
        # P2 입장에서 P1의 트랩 위협을 평가하는 로직으로 발전시킬 수 있습니다.
        return 0.0
    
    def evaluate_line_completion(self, row: int, col: int, piece: Tuple, 
                               board: List[List[int]], pieces: List[Tuple]) -> float:
        """라인 완성 가능성 평가"""
        # TODO: 구현 예정 (이 함수는 evaluate_p1_threat로 대체되었으므로 삭제 또는 주석 처리 예정)
        return 0.0
    
    def backpropagate(self, node, result):
        """결과를 부모 노드들에게 전파"""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1 - result  # 결과 반전 (min-max)
    
    def get_best_move(self, root: Node, is_selection: bool = True):
        """MCTS 실행하여 최선의 수 선택"""
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < self.time_limit:
            node_s = self.select(root)
            node_e = self.expand(node_s) 
            
            sim_node = node_e # P2의 expand는 새 노드 또는 원본 노드를 반환하므로 node_e를 사용
            result = self.simulate(sim_node)
            self.backpropagate(sim_node, result) # 시뮬레이션된 노드로부터 전파
            iterations += 1

        if not root.children:
            # print(f"P2 MCTS Warning: Ran for {iterations} iterations. Root node has no children.")
            # print(f"Root state: board={root.board}, available={root.available_pieces}, move_to_root={root.move}")
            # print(f"Root untried_moves: {root.untried_moves}")
            possible_fallback_moves = list(root.untried_moves)
            if not possible_fallback_moves:
                # print(f"P2 MCTS Error: No children and no untried_moves. Cannot determine best move.")
                if is_selection:
                    # print("P2 MCTS Fallback Error: No piece to select from untried_moves.")
                    return self.available_pieces[0] if self.available_pieces else (0,0,0,0)
                else:
                    # print("P2 MCTS Fallback Error: No position to place from untried_moves.")
                    piece_to_place = root.move if isinstance(root.move, tuple) and len(root.move) == 4 else (0,0,0,0)
                    return (piece_to_place, (0,0)) # P2.place_piece는 (piece,(r,c))에서 (r,c)를 추출
            
            # print("P2 MCTS Fallback: Returning random move from root.untried_moves.")
            return random.choice(possible_fallback_moves)
        
        # 정상적인 경우: 자식 노드 중에서 선택
        if is_selection:
            best_child = min(root.children, key=lambda n: n.wins/n.visits if n.visits > 0 else float('inf'))
        else:
            best_child = max(root.children, key=lambda n: n.wins/n.visits if n.visits > 0 else float('-inf'))
        
        return best_child.move
    
    def select_piece(self):
        """상대방에게 줄 조각 선택"""
        # 루트 노드 초기화: P2가 상대방에게 줄 조각을 선택하는 상황
        root_state = (self.board, tuple(self.available_pieces))
        root = Node(state=root_state, pieces=self.pieces, parent=None, move=None, is_opponent_turn=True)
        return self.get_best_move(root, is_selection=True)

    def place_piece(self, selected_piece):
        """조각을 놓을 위치 선택"""
        current_available_after_selection = list(self.available_pieces)
        # selected_piece는 self.available_pieces에 이미 없어야 정상이지만,
        # Node 초기화를 위해 명시적으로 제거된 상태의 리스트를 만듦.
        if selected_piece in current_available_after_selection:
            current_available_after_selection.remove(selected_piece)
        
        root_state = (self.board, tuple(current_available_after_selection))
        # P2가 조각을 '배치'하는 행위는 P2 자신의 턴(is_opponent_turn=False).
        # 이때 Node의 move 속성은 P2가 받은 조각(selected_piece)임.
        root = Node(state=root_state, pieces=self.pieces, parent=None, move=selected_piece, is_opponent_turn=False)
        
        best_move_action = self.get_best_move(root, is_selection=False)

        # best_move_action은 P2의 expand 함수에서 구성된 (piece_to_place, position_to_place) 형태일 것.
        # 여기서 piece_to_place는 selected_piece와 동일해야 함.
        if isinstance(best_move_action, tuple) and len(best_move_action) == 2 and \
           isinstance(best_move_action[0], tuple) and isinstance(best_move_action[1], tuple) and \
           len(best_move_action[1]) == 2:
            # best_move_action[0]은 piece, best_move_action[1]은 (row, col)
            return best_move_action[1]  # (row, col) 반환
        else:
            # print(f"P2 place_piece: Unexpected move format from get_best_move: {best_move_action}")
            for r_idx in range(4):
                for c_idx in range(4):
                    if self.board[r_idx][c_idx] == 0:
                        return (r_idx, c_idx)
            return (0,0) 

    def get_piece_from_index(self, index: int, pieces: List[Tuple]) -> Optional[Tuple]:
        """인덱스로부터 조각 정보 반환"""
        if index == 0:  # 빈 칸
            return None
        return pieces[index - 1]  # 1-based 인덱스

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
            piece_to_select = move # move가 piece 자체임
            if piece_to_select in new_available_pieces:
                new_available_pieces.remove(piece_to_select)
            # else: 이미 선택된 조각이거나 없는 조각이면 오류지만, MCTS 흐름상 유효한 조각만 올 것으로 가정
        
        return (new_board, tuple(new_available_pieces))

    def place_piece_playout(self, board: List[List[int]], piece_to_place: Tuple, 
                             pieces: List[Tuple]) -> Optional[Tuple[int, int]]:
        """
        AI (P2)가 P1으로부터 받은 `piece_to_place`를 보드에 놓을 위치를 결정하는 휴리스틱.
        방어적이면서 기회를 노리는 전략을 반영.
        :return: 선택된 위치 (row, col) 또는 None (놓을 곳이 없을 때)
        """
        available_locs = [(r, c) for r, c in product(range(4), range(4)) 
                         if board[r][c] == 0]
        if not available_locs:
            return None
            
        best_move_for_p2 = None
        best_score_for_p2 = float('-inf') # P2에게 가장 좋은 수를 찾음

        for r, c in available_locs:
            p2_score_at_loc = 0
            
            # 1. P2가 즉시 이길 수 있는가?
            temp_board_p2_wins = deepcopy(board)
            temp_board_p2_wins[r][c] = self.get_piece_index(piece_to_place, pieces)
            if self.check_win(temp_board_p2_wins, pieces):
                # print(f"P2 Playout: Found winning move at ({r},{c}) for piece {piece_to_place}")
                return (r, c) # 즉시 승리 가능하면 바로 선택
            
            # 2. P1의 잠재적 3목 위협 방어 점수
            defense_bonus = 0
            # (r, c)를 지나는 모든 라인의 좌표 목록 생성
            lines_coords_rc = []
            lines_coords_rc.append([(r, i) for i in range(4)]) # 가로
            lines_coords_rc.append([(i, c) for i in range(4)]) # 세로
            if r == c: lines_coords_rc.append([(i, i) for i in range(4)]) # 주대각선
            if r + c == 3: lines_coords_rc.append([(i, 3-i) for i in range(4)]) # 부대각선

            for line_coords in lines_coords_rc:
                # 원본 보드에서 해당 라인의 조각 인덱스 가져오기
                original_line_indices = [board[rr][cc] for rr, cc in line_coords]
                
                # 라인에 빈칸이 정확히 하나였고, 그 위치가 현재 (r, c)인지 확인
                if original_line_indices.count(0) == 1 and board[r][c] == 0: 
                    # 원본 라인에서 실제 조각들 가져오기 (빈칸 제외)
                    actual_pieces_orig = [self.get_piece_from_index(idx, pieces) for idx in original_line_indices if idx != 0]
                    
                    if len(actual_pieces_orig) == 3:
                        # 이 3개의 조각이 P1의 3목 위협을 형성하는지 확인 (모두 같은 속성)
                        for dim in range(4):
                            if not actual_pieces_orig: continue # 방어 코드
                            first_val = actual_pieces_orig[0][dim]
                            is_p1_threat_line = all(p[dim] == first_val for p in actual_pieces_orig)
                            
                            if is_p1_threat_line:
                                # P1의 3목 위협 라인이었다. P2가 놓는 조각이 이 라인을 방해하는가?
                                if piece_to_place[dim] != first_val:
                                    defense_bonus += 2.0 # P1의 3목 위협을 막는 것에 높은 점수 부여
                                    # print(f"P2 Playout: Placing at ({r},{c}) blocks P1 3-threat (dim {dim}) for piece {piece_to_place}")
                                    break # 해당 라인에 대한 다른 차원 검사 불필요
            p2_score_at_loc += defense_bonus * 2.0 # 방어 점수에 가중치 부여

            # 3. P2에게 유리한 라인 형성 점수
            p2_line_completion_score = self.evaluate_p2_line_completion(r, c, piece_to_place, board, pieces)
            p2_score_at_loc += p2_line_completion_score * 1.0

            # 4. 중앙 선호 등 기타 휴리스틱
            center_dist = abs(r - 1.5) + abs(c - 1.5)
            p2_score_at_loc += (4 - center_dist) * 0.1

            if p2_score_at_loc > best_score_for_p2:
                best_score_for_p2 = p2_score_at_loc
                best_move_for_p2 = (r, c)
        
        # 최선의 수가 없으면 랜덤 선택
        if best_move_for_p2 is None:
             # print("P2 Playout: No best move found, choosing random.")
             best_move_for_p2 = available_locs[np.random.randint(len(available_locs))]
             
        # print(f"P2 Playout: Chose ({best_move_for_p2}) for piece {piece_to_place} with score {best_score_for_p2}")
        return best_move_for_p2

    def p1_select_piece_playout(self, board: List[List[int]], available_pieces: List[Tuple], 
                                all_pieces_info: List[Tuple]) -> Optional[Tuple]:
        """
        시뮬레이션 중 P1이 P2에게 줄 조각을 선택하는 휴리스틱.
        P1은 공격적이므로 P2에게 가장 위협적인(트랩 점수가 높은) 조각을 주려고 시도.
        :return: P1이 선택한 조각 또는 None
        """
        if not available_pieces:
            return None

        best_piece_for_p1_to_give = None
        highest_threat_score_to_p2 = -1.0

        for piece_option in available_pieces:
            # P1이 이 piece_option을 P2에게 주었을 때, P2에게 얼마나 트랩이 되는지 평가
            threat_score = self.evaluate_p1_trap_threat(piece_option, board, all_pieces_info)
            # TODO: P1 조각 선택 시뮬레이션 휴리스틱 개선 (단순히 트랩 위협 외 P1 관점의 다른 요소 고려)
            if threat_score > highest_threat_score_to_p2:
                highest_threat_score_to_p2 = threat_score
                best_piece_for_p1_to_give = piece_option

        # 가장 위협적인 조각을 찾지 못했거나 모든 조각의 점수가 동일하게 낮으면 랜덤 선택
        if best_piece_for_p1_to_give is None and available_pieces:
            return random.choice(available_pieces)
        return best_piece_for_p1_to_give

    def evaluate_p2_line_completion(self, row: int, col: int, piece: Tuple,
                                   board: List[List[int]], pieces: List[Tuple]) -> float:
        """P2 관점에서 라인 완성 가능성 평가 (P1.evaluate_line_completion 기반)."""
        score = 0.0
        temp_board = deepcopy(board)
        temp_board[row][col] = self.get_piece_index(piece, pieces) # P2의 get_piece_index 사용

        # 가로/세로 라인 평가
        lines_to_check = [
            temp_board[row],  # 가로
            [temp_board[i][col] for i in range(4)]  # 세로
        ]
        # 대각선 추가
        if row == col:
            lines_to_check.append([temp_board[i][i] for i in range(4)]) # 주대각선
        if row + col == 3:
            lines_to_check.append([temp_board[i][3-i] for i in range(4)]) # 부대각선
        
        for line_indices in lines_to_check:
            if len(line_indices) == 0: # 혹시 모를 빈 대각선 리스트 처리 (수정됨)
                continue
            for dim in range(4):
                # P2의 _count_same_attribute_pieces 사용 (이 함수가 P2에 정의되어 있다고 가정)
                count = self._count_same_attribute_pieces(line_indices, pieces, dim)
                if count == 4:  # P2 승리 라인
                    score += 10.0
                elif count == 3:  # P2의 잠재적 승리 (3개 완성)
                    score += 3.0
                elif count == 2:  # P2의 발전 가능성 (2개 완성)
                    score += 0.5
        
        # 2x2 부분 격자 평가 (P1의 evaluate_line_completion 로직 참조)
        # (row, col)을 포함하는 모든 2x2 격자 검사
        for r_start_offset in [-1, 0]: # 2x2의 좌상단 r좌표를 기준으로 현재 놓은 위치와의 상대적 위치
            for c_start_offset in [-1, 0]: # 2x2의 좌상단 c좌표를 기준으로 현재 놓은 위치와의 상대적 위치
                r_start = row + r_start_offset
                c_start = col + c_start_offset
                
                # 유효한 2x2 좌상단 좌표인지 확인 (0,0) (0,1) (0,2) / (1,0) ... (2,2)
                if 0 <= r_start <= 2 and 0 <= c_start <= 2:
                    # 현재 piece가 놓인 (row,col)이 이 2x2 격자 (r_start,c_start)의 일부인지 확인
                    if not (r_start <= row < r_start + 2 and c_start <= col < c_start + 2):
                        continue

                    subgrid_indices = [
                        temp_board[r_start][c_start], temp_board[r_start][c_start+1],
                        temp_board[r_start+1][c_start], temp_board[r_start+1][c_start+1]
                    ]
                    
                    for dim in range(4):
                        count = self._count_same_attribute_pieces(subgrid_indices, pieces, dim)
                        if count == 4:  # P2의 2x2 승리
                            score += 10.0 
                        elif count == 3:  # P2의 잠재적 2x2 (3개 완성)
                            score += 3.0 
                        elif count == 2: # P2의 2x2 발전 가능성 (2개 완성)
                             score += 0.5
        return score