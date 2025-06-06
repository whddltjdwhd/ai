from typing import List, Tuple, Optional
from functools import lru_cache

class P1:
    # 게임 평가 상수 (P1 관점)
    WIN_SCORE = 10000.0
    LOSE_SCORE = -10000.0
    DRAW_SCORE = 0.0
    FORK_BONUS = 600.0  # 양방 3목 기회 (선공이므로 더 높게 설정)
    MATCHING_ATTRIBUTES_BONUS = 15.0  # 3개 속성 일치 라인 하나당 보너스 (선공이므로 더 높게 설정)
    CENTER_BONUS = 30.0  # 중앙 위치 보너스 (선공이므로 더 높게 설정)
    CORNER_BONUS = 15.0  # 코너 위치 보너스 (선공이므로 더 높게 설정)
    IMMEDIATE_WIN_BONUS = 1200.0  # 즉시 승리 기회 (선공이므로 더 높게 설정)
    THREE_IN_ROW_BONUS = 120.0  # 3목 기회 (선공이므로 더 높게 설정)
    FIRST_MOVE_BONUS = 50.0  # 첫 수 보너스 (선공 특성 반영)

    # P1 관점의 위험도 상수 (높을수록 위험)
    FORK_DANGER_SCORE = 1200.0        # 양방 3목 필승 위협 (선공이므로 더 높게 설정)
    THREE_IN_ROW_DANGER = 120.0       # 단일 3목 위협 (선공이므로 더 높게 설정)
    OPPONENT_PIECE_ADVANTAGE_DANGER = 60.0  # 상대방에게 유리한 피스를 주는 위험 (선공이므로 더 높게 설정)
    CONSECUTIVE_THREAT_DANGER = 250.0  # 연속된 위협 위험도 (선공이므로 더 높게 설정)
    BLOCK_FORK_DANGER = 900.0         # 2x2 블록 포크 위험도 (선공이므로 더 높게 설정)
    CENTER_THREE_IN_ROW_DANGER = 180.0  # 중앙 위치 3목 위협 (선공이므로 더 높게 설정)
    CORNER_THREE_IN_ROW_DANGER = 90.0  # 코너 위치 3목 위협 (선공이므로 더 높게 설정)
    EARLY_GAME_DANGER = 100.0  # 초반 위험도 (선공 특성 반영)

    def __init__(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]]):
        # board가 튜플인 경우 리스트로 변환
        self.board = [list(row) for row in board] if isinstance(board, tuple) else [row.copy() for row in board]
        # available_pieces가 튜플인 경우 리스트로 변환
        self.available_pieces = list(available_pieces) if isinstance(available_pieces, tuple) else available_pieces.copy()
        # 모든 16개 피스 조합 생성 및 인덱스 매핑
        self.pieces = [(i,j,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.piece_to_index = {p: idx+1 for idx,p in enumerate(self.pieces)}
        self.index_to_piece = {idx+1: p for idx,p in enumerate(self.pieces)}  # 인덱스로 피스 찾기 추가
        self.minimax_depth = self._get_minimax_depth()  # 동적 깊이 설정
        self.chosen_piece = None  # place_piece에서 결정된 '상대에게 줄 피스'를 저장할 변수

    def _get_minimax_depth(self) -> int:
        """
        게임 상태에 따라 적절한 Minimax 깊이를 반환
        Returns:
            int: Minimax 탐색 깊이
        """
        empty_count = sum(1 for row in self.board for cell in row if cell == 0)
        
        # 게임 초반 (12개 이상 빈칸)
        if empty_count >= 14:
            return 1  # 초반에도 적당한 깊이로 탐색
        
        # 게임 중반 (8-11개 빈칸)
        elif empty_count >= 8:
            return 2  # 중반에는 좀 더 깊이 탐색
        
        # 게임 후반 (4-7개 빈칸)
        elif empty_count >= 4:
            return 3  # 후반에는 더 깊이 탐색
        
        # 게임 막바지 (3개 이하 빈칸)
        else:
            return 4  # 막바지에는 최대한 깊게 탐색

    def _danger_score(self, board: List[List[int]], piece: Tuple[int,int,int,int]) -> int:
        """
        해당 피스가 얼마나 위험한지 점수 계산 (P2의 관점: P1이 이 피스를 받았을 때)
        Returns:
            int: 점수가 높을수록 위험한 피스 (P1에게 유리한 피스)
        """
        # board가 튜플인 경우 리스트로 변환
        board_list = [list(row) for row in board] if isinstance(board, tuple) else board
        score = 0
        
        # 1. P1이 이 피스로 즉시 승리할 수 있는지 체크 (최고 위험도)
        for r in range(4):
            for c in range(4):
                if board_list[r][c] == 0:
                    if self._is_winning_move(board_list, r, c, piece):
                        score += self.WIN_SCORE # P1이 승리하면 매우 높은 위험
                        return score # 즉시 반환 (더 이상 계산 불필요)

        # 2. P1이 이 피스로 양방 3목 필승을 만들 수 있는지 체크 (높은 위험도)
        if self._is_unavoidable_fork(board_list, piece):
            score += self.FORK_DANGER_SCORE
            
        # 3. P1이 이 피스로 3목 기회를 얻을 수 있는지 체크 (중간 위험도)
        # 위치에 따른 가중치 차등 적용
        for r in range(4):
            for c in range(4):
                if board_list[r][c] == 0:
                    matching_attributes = self._count_matching_attributes(board_list, r, c, piece)
                    if matching_attributes >= 1:
                        score += self.THREE_IN_ROW_DANGER * matching_attributes
                        # 중앙/코너 위치에 따른 3목 위협 가중치
                        if (r in [1, 2] and c in [1, 2]): # 중앙 4칸
                            score += self.CENTER_THREE_IN_ROW_DANGER * matching_attributes
                        elif (r in [0, 3] and c in [0, 3]): # 코너 4칸
                            score += self.CORNER_THREE_IN_ROW_DANGER * matching_attributes

        # 4. P1이 이 피스로 2x2 블록 포크 기회를 얻을 수 있는지 체크
        for r in range(3):
            for c in range(3):
                # 해당 2x2 블록에 빈 칸이 있고 P1이 놓을 피스가 블록에 포함될 경우
                current_block_indices = [board_list[r][c], board_list[r][c+1], board_list[r+1][c], board_list[r+1][c+1]]
                if 0 in current_block_indices:
                    # 가상으로 피스를 놓아보고 2x2 포크 기회 확인
                    for br in [r, r+1]:
                        for bc in [c, c+1]:
                            if board_list[br][bc] == 0:
                                temp_board = [row.copy() for row in board_list]
                                temp_board[br][bc] = self.piece_to_index[piece]
                                # P1이 해당 2x2 블록으로 4목을 만들 수 있는 기회가 2개 이상이면 포크
                                block_fork_count = 0
                                for attr_idx in range(4):
                                    # 해당 2x2 블록에 놓았을 때 3개 이상의 같은 속성을 가진 피스가 두 가지 이상의 방식으로 생성되는지 확인
                                    sub_block_pieces = [self.index_to_piece[idx] for idx in current_block_indices if idx != 0]
                                    if len(sub_block_pieces) == 3:
                                        if sum(1 for p in sub_block_pieces if p[attr_idx] == piece[attr_idx]) == 3:
                                            block_fork_count += 1
                
                if block_fork_count >= 2:
                    score += self.BLOCK_FORK_DANGER * block_fork_count

        # 5. 연속된 위협 감지 및 평가
        potential_threat_lines = 0
        for r in range(4):
            for c in range(4):
                if board_list[r][c] == 0:
                    temp_board = [row.copy() for row in board_list]
                    temp_board[r][c] = self.piece_to_index[piece]
                    if self._count_matching_attributes(temp_board, r, c, piece) >= 1:
                        potential_threat_lines += 1
        if potential_threat_lines >= 2: # 2개 이상의 잠재적 3목 라인 형성
            score += self.CONSECUTIVE_THREAT_DANGER
        
        return score
    
    def place_piece(self, selected_piece: Tuple[int,int,int,int]) -> Tuple[int,int]:
        """
        P1이 Minimax를 사용하여 최적의 위치를 결정합니다.
        P1은 항상 선공이므로, Minimax 탐색 시 최대화 플레이어의 관점에서 시작합니다.
        Args:
            selected_piece: P2가 P1에게 준, P1이 보드에 놓을 피스
        Returns:
            Tuple[int, int]: 피스를 놓을 위치 (r, c)
        """
        
        # available_pieces에서 selected_piece 제거 (main.py 동기화)
        if selected_piece in self.available_pieces:
            self.available_pieces.remove(selected_piece)
        
        # 매 턴마다 깊이 업데이트 및 캐시 초기화
        self.minimax_depth = self._get_minimax_depth()
        self.minimax.cache_clear()
        self._evaluate_game_state.cache_clear()
        
        # Minimax 호출 (P1은 항상 선공이므로 is_maximizing_player=True)
        eval_score, _, best_pos_to_place = self.minimax(
            tuple(tuple(r) for r in self.board),  # 현재 보드 상태 (튜플)
            selected_piece,  # P1이 놓을 피스 (P2가 P1에게 준 피스)
            tuple(self.available_pieces),  # P1이 P2에게 줄 수 있는 피스 목록
            self.minimax_depth,  # 동적으로 조정된 깊이 사용
            float('-inf'), 
            float('inf'),
            True  # P1은 최대화 플레이어
        )
        
        # Minimax가 최적의 위치를 선택하지 못한 경우
        if best_pos_to_place is None:
            print("Minimax가 최적의 위치를 선택하지 못했습니다.")
            # --- 대체 로직: 놓을 위치 선택 ---
            empty_positions = self._get_empty_positions(self.board)
            if not empty_positions:
                return (0, 0)
            
            position_scores = []
            for r, c in empty_positions:
                # P1의 관점에서 해당 위치에 selected_piece를 놓았을 때의 점수를 평가
                score_at_pos = self._evaluate_position(tuple(tuple(row) for row in self.board), (r, c), selected_piece)
                position_scores.append((score_at_pos, (r, c)))
            
            # P1은 자신의 점수를 최대화해야 하므로 내림차순 정렬
            position_scores.sort(key=lambda x: x[0], reverse=True)
            best_pos_to_place = position_scores[0][1]
        
        # 보드 상태 업데이트
        self.board[best_pos_to_place[0]][best_pos_to_place[1]] = self.piece_to_index[selected_piece]
        
        return best_pos_to_place

    def select_piece(self) -> Tuple[int,int,int,int]:
        """
        P1이 Minimax를 사용하여 상대에게 줄 최적의 피스를 결정합니다.
        Returns:
            Tuple[int,int,int,int]: 상대에게 줄 최적의 피스
        """
        
        self.minimax_depth = self._get_minimax_depth()  # 매 턴마다 깊이 업데이트
        # Minimax 호출 전에 캐시 초기화
        self.minimax.cache_clear()
        self._evaluate_game_state.cache_clear()
        
        # Minimax 호출 (P1은 항상 선공이므로 is_maximizing_player=True)
        eval_score, best_piece_to_give, _ = self.minimax(
            tuple(tuple(r) for r in self.board),  # 현재 보드 상태 (튜플)
            None,  # 현재 턴에는 놓을 피스가 없음
            tuple(self.available_pieces),  # P1이 P2에게 줄 수 있는 피스 목록
            self.minimax_depth,  # 동적으로 조정된 깊이 사용
            float('-inf'), 
            float('inf'),
            True  # P1은 최대화 플레이어
        )
        
        # Minimax가 최적의 피스를 선택하지 못한 경우
        if best_piece_to_give is None:
            print("Minimax가 최적의 피스를 선택하지 못했습니다.")
            # 위험도가 가장 낮은 피스 선택
            piece_danger_scores = []
            for piece_candidate in self.available_pieces:
                danger = self._danger_score(
                    tuple(tuple(r) for r in self.board),  # 현재 보드
                    piece_candidate  # P2에게 줄 피스 후보
                )
                piece_danger_scores.append((danger, piece_candidate))
            
            # 위험도가 낮은 순으로 정렬 (오름차순)
            piece_danger_scores.sort(key=lambda x: x[0])
            best_piece_to_give = piece_danger_scores[0][1]  # 가장 위험도가 낮은 피스 선택
                
        # 선택한 피스를 available_pieces에서 제거
        if best_piece_to_give in self.available_pieces:
            self.available_pieces.remove(best_piece_to_give)
        
        return best_piece_to_give

    def _evaluate_position(self, board: List[List[int]], pos: Optional[Tuple[int, int]] = None, piece: Optional[Tuple[int,int,int,int]] = None) -> float:
        """
        개선된 위치 평가 함수
        Args:
            board: 현재 게임 보드
            pos: 평가할 위치 (선택적)
            piece: 평가할 피스 (선택적)
        Returns:
            float: 평가 점수
        """
        score = 0
        
        # 위치와 피스가 주어진 경우에만 해당 평가 수행
        if pos is not None and piece is not None:
            # 1. 중앙 위치 가중치
            if 1 <= pos[0] <= 2 and 1 <= pos[1] <= 2:
                score += 5
            
            # 2. 모서리 위치 가중치
            if (pos[0] in [0, 3] and pos[1] in [0, 3]):
                score += 2
            
            # 3. 기존 피스와의 관계 평가
            for r in range(4):
                for c in range(4):
                    if board[r][c] != 0:
                        existing_piece = self.pieces[board[r][c]-1]
                        # 같은 속성을 가진 피스 근처에 놓으면 가중치 부여
                        matching_attrs = sum(1 for i in range(4) if piece[i] == existing_piece[i])
                        score += matching_attrs * 2
            
            # 4. 승리 가능성 평가
            if self._is_winning_move(board, pos[0], pos[1], piece):
                score += 1000
            elif self._has_fork_opportunity(board, self.available_pieces, piece, pos):
                score += 500
        
        # 5. 전체 보드 상태 평가
        # 중앙 제어 점수
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        for r, c in center_positions:
            if board[r][c] != 0:
                score += (1 if board[r][c] % 2 == 0 else -1) * self.CENTER_BONUS
        
        # 6. 잠재적 라인 평가
        potential_lines = self._count_potential_lines(board)
        score += potential_lines * 10
        
        return score

    @lru_cache(maxsize=16384)
    def minimax(self, 
                board: Tuple[Tuple[int, ...], ...], 
                current_player_piece: Optional[Tuple[int,int,int,int]], 
                remaining_available_pieces: Tuple[Tuple[int,int,int,int], ...], 
                depth: int, 
                alpha: float, 
                beta: float,
                is_maximizing_player: bool) -> Tuple[float, Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int]]]:
        """
        Minimax 알고리즘 구현 (P2는 항상 후공)
        Args:
            board: 현재 게임 보드 상태
            current_player_piece: 현재 플레이어가 놓을 피스 (None일 수 있음)
            remaining_available_pieces: 남은 사용 가능한 피스 목록
            depth: 현재 탐색 깊이
            alpha: 알파 값 (최대화 플레이어의 최선의 값)
            beta: 베타 값 (최소화 플레이어의 최선의 값)
            is_maximizing_player: True면 P1(최대화), False면 P2(최소화)의 턴
        Returns:
            Tuple[float, Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int]]]: 
            (평가 점수, 상대에게 줄 피스, 놓을 위치)
        """
        # 보드가 가득 찬 경우
        if all(cell != 0 for row in board for cell in row):
            eval_score = self._evaluate_game_state(board)
            return (eval_score, 
                   remaining_available_pieces[0] if remaining_available_pieces else None,
                   (0, 0))  # 보드가 가득 찬 경우 (0,0) 반환

        # 게임 종료 체크 (피스 선택 단계에서는 체크하지 않음)
        if current_player_piece is not None and self._check_win_cached(board):
            return (float('-inf') if is_maximizing_player else float('inf'), 
                   remaining_available_pieces[0] if remaining_available_pieces else None,
                   (0, 0))  # 승리한 경우 (0,0) 반환

        # 깊이 제한 또는 사용 가능한 피스가 없는 경우
        if depth == 0 or not remaining_available_pieces:
            eval_score = self._evaluate_game_state(board)
            empty_positions = self._get_empty_positions(list(map(list, board)))
            if not empty_positions:
                return (eval_score,
                       remaining_available_pieces[0] if remaining_available_pieces else None,
                       (0, 0))  # 빈 위치가 없는 경우 (0,0) 반환
            return (eval_score,
                   remaining_available_pieces[0] if remaining_available_pieces else None,
                   empty_positions[0])  # 첫 번째 빈 위치 반환

        if is_maximizing_player:  # P1의 턴
            max_eval = float('-inf')
            best_pos_to_place = None
            best_piece_to_give = None

            # 정렬된 이동 목록 가져오기
            ordered_moves = self._get_ordered_moves(
                list(map(list, board)),
                list(remaining_available_pieces),
                current_player_piece
            )

            if not ordered_moves:  # 이동 가능한 수가 없는 경우
                empty_positions = self._get_empty_positions(list(map(list, board)))
                if empty_positions:
                    return (self._evaluate_game_state(board),
                           remaining_available_pieces[0] if remaining_available_pieces else None,
                           empty_positions[0])
                return (self._evaluate_game_state(board),
                       remaining_available_pieces[0] if remaining_available_pieces else None,
                       (0, 0))

            skipped_pieces = []
            for r, c, piece_to_give in ordered_moves:
                if current_player_piece is not None:  # 피스를 놓는 턴인 경우
                    temp_board_after_place = [list(row) for row in board]
                    temp_board_after_place[r][c] = self.piece_to_index[current_player_piece]
                    
                    # 내가 이 수로 승리하는지 체크
                    if self._check_win(temp_board_after_place):
                        return (float('inf'), piece_to_give, (r,c))

                    # 상대가 이 피스로 이길 수 있는지 체크
                    if self._is_immediate_win_for_opponent(temp_board_after_place, list(remaining_available_pieces), piece_to_give):
                        skipped_pieces.append(piece_to_give)
                        continue

                    next_available_pieces = tuple(p for p in remaining_available_pieces if p != piece_to_give)

                    # 재귀 호출 (P2의 턴)
                    eval_score, _, _ = self.minimax(
                        tuple(tuple(r_sub) for r_sub in temp_board_after_place),
                        piece_to_give,
                        next_available_pieces,
                        depth - 1,
                        alpha,
                        beta,
                        False
                    )
                else:  # 피스를 선택하는 턴인 경우
                    next_available_pieces = tuple(p for p in remaining_available_pieces if p != piece_to_give)
                    eval_score, _, _ = self.minimax(
                        board,
                        piece_to_give,
                        next_available_pieces,
                        depth - 1,
                        alpha,
                        beta,
                        False
                    )

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_pos_to_place = (r, c) if current_player_piece is not None else None
                    best_piece_to_give = piece_to_give

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            # 모든 수가 skipped_pieces에 있는 경우
            if best_pos_to_place is None and current_player_piece is not None:
                empty_positions = self._get_empty_positions(list(map(list, board)))
                if empty_positions:
                    return (self._evaluate_game_state(board),
                           remaining_available_pieces[0] if remaining_available_pieces else None,
                           empty_positions[0])
                return (self._evaluate_game_state(board),
                       remaining_available_pieces[0] if remaining_available_pieces else None,
                       (0, 0))

            return (max_eval, best_piece_to_give, best_pos_to_place)

        else:  # P2의 턴
            min_eval = float('inf')
            best_pos_to_place = None
            best_piece_to_give = None

            # 피스 선택 단계에서는 모든 가능한 피스를 평가
            if current_player_piece is None:
                skipped_pieces = []
                for piece_to_give in remaining_available_pieces:
                    next_available_pieces = tuple(p for p in remaining_available_pieces if p != piece_to_give)
                    eval_score, _, _ = self.minimax(
                        board,
                        piece_to_give,
                        next_available_pieces,
                        depth - 1,
                        alpha,
                        beta,
                        True
                    )

                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_piece_to_give = piece_to_give

                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break

                if best_piece_to_give is None and remaining_available_pieces:
                    best_piece_to_give = remaining_available_pieces[0]

                return (min_eval, best_piece_to_give, None)

            # 피스 배치 단계
            ordered_moves = self._get_ordered_moves(
                list(map(list, board)),
                list(remaining_available_pieces),
                current_player_piece
            )

            if not ordered_moves:  # 이동 가능한 수가 없는 경우
                empty_positions = self._get_empty_positions(list(map(list, board)))
                if empty_positions:
                    return (self._evaluate_game_state(board),
                           remaining_available_pieces[0] if remaining_available_pieces else None,
                           empty_positions[0])
                return (self._evaluate_game_state(board),
                       remaining_available_pieces[0] if remaining_available_pieces else None,
                       (0, 0))

            skipped_pieces = []
            for r, c, piece_to_give in ordered_moves:
                temp_board_after_place = [list(row) for row in board]
                temp_board_after_place[r][c] = self.piece_to_index[current_player_piece]

                if self._check_win(temp_board_after_place):
                    return (float('-inf'), piece_to_give, (r,c))

                if self._is_immediate_win_for_opponent(temp_board_after_place, list(remaining_available_pieces), piece_to_give):
                    skipped_pieces.append(piece_to_give)
                    continue

                next_available_pieces = tuple(p for p in remaining_available_pieces if p != piece_to_give)

                # 재귀 호출 (P1의 턴)
                eval_score, _, _ = self.minimax(
                    tuple(tuple(r_sub) for r_sub in temp_board_after_place),
                    piece_to_give,
                    next_available_pieces,
                    depth - 1,
                    alpha,
                    beta,
                    True
                )

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_pos_to_place = (r, c)
                    best_piece_to_give = piece_to_give

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            # 모든 수가 skipped_pieces에 있는 경우
            if best_pos_to_place is None:
                empty_positions = self._get_empty_positions(list(map(list, board)))
                if empty_positions:
                    return (self._evaluate_game_state(board),
                           remaining_available_pieces[0] if remaining_available_pieces else None,
                           empty_positions[0])
                return (self._evaluate_game_state(board),
                       remaining_available_pieces[0] if remaining_available_pieces else None,
                       (0, 0))

            return (min_eval, best_piece_to_give, best_pos_to_place)

    def _get_ordered_moves(self, board: List[List[int]], current_available_pieces: List[Tuple[int,int,int,int]], piece_to_place: Optional[Tuple[int,int,int,int]] = None) -> List[Tuple[int, int, Tuple[int,int,int,int]]]:
        """
        이동 순서를 최적화하여 반환
        Args:
            board: 현재 게임 보드
            current_available_pieces: 현재 사용 가능한 피스 목록
            piece_to_place: 현재 턴에 보드에 놓을 피스 (선택적)
        Returns:
            List[Tuple[int, int, Tuple[int,int,int,int]]]: (r, c, piece) 튜플 리스트
        """
        moves = []
        # board가 튜플인 경우 리스트로 변환
        board_list = [list(row) for row in board] if isinstance(board, tuple) else board
        empty_positions = self._get_empty_positions(board_list)
        
        # 1. 승리 가능한 수를 먼저 평가
        if piece_to_place:
            for r, c in empty_positions:
                if self._is_winning_move(board_list, r, c, piece_to_place):
                    return [(r, c, piece_to_place)]  # 승리 수가 있으면 즉시 반환
        
        # 2. 상대방의 즉시 승리를 막는 수를 다음으로 평가
        blocking_moves = []
        for r, c in empty_positions:
            for piece in current_available_pieces:
                temp_board = [row.copy() for row in board_list]
                temp_board[r][c] = self.piece_to_index[piece]
                
                # 이 위치에 피스를 놓음으로써 상대방의 즉시 승리 기회를 막을 수 있는지 체크
                blocks_win = False
                for opp_piece in current_available_pieces:
                    if opp_piece != piece:
                        for opp_r, opp_c in empty_positions:
                            if (opp_r, opp_c) != (r, c):
                                if self._is_winning_move(temp_board, opp_r, opp_c, opp_piece):
                                    blocks_win = True
                                    break
                        if blocks_win:
                            break
                
                if blocks_win:
                    blocking_moves.append((r, c, piece))
        
        if blocking_moves:
            moves.extend(blocking_moves)
        
        # 3. 양방 3목 기회가 있는 수를 다음으로 평가
        fork_moves = []
        for r, c in empty_positions:
            for piece in current_available_pieces:
                if (r, c, piece) not in moves and self._has_fork_opportunity(board_list, current_available_pieces, piece, (r, c)):
                    fork_moves.append((r, c, piece))
        
        if fork_moves:
            moves.extend(fork_moves)
        
        # 4. 나머지 수들을 평가 점수 순으로 정렬
        remaining_moves = []
        for r, c in empty_positions:
            for piece in current_available_pieces:
                if (r, c, piece) not in moves:
                    score = self._evaluate_position(board_list, (r, c), piece)
                    remaining_moves.append((score, r, c, piece))
        
        # 평가 점수 순으로 정렬 (최대화 플레이어는 높은 점수부터, 최소화 플레이어는 낮은 점수부터)
        remaining_moves.sort(reverse=True)
        moves.extend([(r, c, piece) for _, r, c, piece in remaining_moves])
        
        return moves

    def _get_empty_positions(self, board: List[List[int]]) -> List[Tuple[int,int]]:
        return [(r,c) for r in range(4) for c in range(4) if board[r][c] == 0]

    @lru_cache(maxsize=1024)
    def _check_win_cached(self, board_tuple: Tuple[Tuple[int, ...], ...]) -> bool:
        return self._check_win(list(map(list, board_tuple)))

    def _check_win(self, board: List[List[int]]) -> bool:
        # 가로/세로/대각선
        for i in range(4):
            if self._check_line([board[i][j] for j in range(4)]) or \
               self._check_line([board[j][i] for j in range(4)]):
                return True
        # 대각선
        if self._check_line([board[i][i] for i in range(4)]) or \
           self._check_line([board[i][3-i] for i in range(4)]):
            return True
        # 2x2 블록
        for r in range(3):
            for c in range(3):
                sub = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in sub and self._check_2x2(sub):
                    return True
        return False

    def _check_line(self, line: List[int]) -> bool:
        """
        주어진 라인(4개 피스 인덱스)이 쿼터 게임의 승리 조건을 만족하는지 확인합니다.
        쿼터 승리 조건: 4개의 피스가 하나 이상의 속성을 공유.
        """
        filled_indices = [idx for idx in line if idx != 0]
        if len(filled_indices) < 4: return False  # 4개의 피스가 모두 채워져야 함

        attrs = [self.pieces[idx-1] for idx in filled_indices]
        
        # 4개의 피스가 하나 이상의 속성을 공유하는지 확인
        for i in range(4):  # 각 속성 (0:크기, 1:모양, 2:색깔, 3:구멍)
            if len(set(attr[i] for attr in attrs)) == 1:  # 모든 피스가 해당 속성에서 동일한 값을 가지는지
                return True
        return False

    def _check_2x2(self, block: List[int]) -> bool:
        """
        주어진 2x2 블록(4개 피스 인덱스)이 쿼터 게임의 승리 조건을 만족하는지 확인합니다.
        쿼터 2x2 승리 조건: 4개의 피스가 하나 이상의 속성을 공유.
        """
        if block.count(0) > 0: return False  # 2x2 블록에 빈 칸이 있으면 안 됨
        attrs = [self.pieces[idx-1] for idx in block]
        
        # 4개의 피스가 하나 이상의 속성을 공유하는지 확인
        for i in range(4):  # 각 속성 (0:크기, 1:모양, 2:색깔, 3:구멍)
            if len(set(attr[i] for attr in attrs)) == 1:  # 모든 피스가 해당 속성에서 동일한 값을 가지는지
                return True
        return False

    def _count_potential_lines(self, board: List[List[int]]) -> int:
        """
        잠재적인 승리 라인의 수를 계산
        """
        potential_lines = 0
        
        # 1. 가로/세로 라인 체크
        for i in range(4):
            # 가로 라인
            row_pieces = [self.pieces[board[i][j]-1] for j in range(4) if board[i][j] != 0]
            if len(row_pieces) >= 2:  # 2개 이상의 피스가 있는 라인만 체크
                for attr in range(4):
                    matches = sum(1 for p in row_pieces if p[attr] == row_pieces[0][attr])
                    if matches >= 2:  # 2개 이상의 피스가 같은 속성을 공유
                        potential_lines += 1
            
            # 세로 라인
            col_pieces = [self.pieces[board[j][i]-1] for j in range(4) if board[j][i] != 0]
            if len(col_pieces) >= 2:
                for attr in range(4):
                    matches = sum(1 for p in col_pieces if p[attr] == col_pieces[0][attr])
                    if matches >= 2:
                        potential_lines += 1
        
        # 2. 대각선 체크
        # 주 대각선
        main_diag_pieces = [self.pieces[board[i][i]-1] for i in range(4) if board[i][i] != 0]
        if len(main_diag_pieces) >= 2:
            for attr in range(4):
                matches = sum(1 for p in main_diag_pieces if p[attr] == main_diag_pieces[0][attr])
                if matches >= 2:
                    potential_lines += 1
        
        # 부 대각선
        anti_diag_pieces = [self.pieces[board[i][3-i]-1] for i in range(4) if board[i][3-i] != 0]
        if len(anti_diag_pieces) >= 2:
            for attr in range(4):
                matches = sum(1 for p in anti_diag_pieces if p[attr] == anti_diag_pieces[0][attr])
                if matches >= 2:
                    potential_lines += 1
        
        # 3. 2x2 블록 체크
        for r in range(3):
            for c in range(3):
                block = [
                    board[r][c], board[r][c+1],
                    board[r+1][c], board[r+1][c+1]
                ]
                if block.count(0) == 0:  # 빈칸이 없는 경우만 체크
                    block_pieces = [self.pieces[idx-1] for idx in block]
                    if len(block_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
                        for i in range(4):
                            if len(set(p[i] for p in block_pieces)) == 1:
                                potential_lines += 1
        
        return potential_lines

    def _get_line_pieces(self, board: List[List[int]], line_type: str, index: int) -> List[Tuple[int,int,int,int]]:
        """
        보드에서 특정 라인(가로/세로/대각선)의 피스들을 가져옴
        Args:
            board: 게임 보드
            line_type: 'row', 'col', 'main_diag', 'anti_diag' 중 하나
            index: 라인 인덱스 (가로/세로의 경우 0-3, 대각선의 경우 무시)
        Returns:
            해당 라인의 피스 리스트
        """
        pieces = []
        if line_type == 'row':
            pieces = [self.pieces[board[index][j]-1] for j in range(4) if board[index][j] != 0]
        elif line_type == 'col':
            pieces = [self.pieces[board[j][index]-1] for j in range(4) if board[j][index] != 0]
        elif line_type == 'main_diag':
            pieces = [self.pieces[board[i][i]-1] for i in range(4) if board[i][i] != 0]
        elif line_type == 'anti_diag':
            pieces = [self.pieces[board[i][3-i]-1] for i in range(4) if board[i][3-i] != 0]
        return pieces

    def _check_line_win(self, pieces: List[Tuple[int,int,int,int]]) -> bool:
        """
        주어진 피스들이 승리 조건을 만족하는지 체크
        """
        if len(pieces) < 3:  # 3개 미만이면 체크할 필요 없음
            return False
        
        for i in range(4):  # 각 속성에 대해
            if len(set(p[i] for p in pieces)) == 1:
                return True
        return False

    def _is_winning_move(self, board: List[List[int]], r: int, c: int, piece: Tuple[int,int,int,int]) -> bool:
        """
        특정 위치에 특정 피스를 놓았을 때 승리하는지 확인
        """
        piece_idx = self.piece_to_index[piece]
        # board가 튜플인 경우 리스트로 변환
        board_list = [list(row) for row in board] if isinstance(board, tuple) else board
        temp_board = [row.copy() for row in board_list]
        temp_board[r][c] = piece_idx
        
        # 가로/세로 체크
        row_pieces = self._get_line_pieces(temp_board, 'row', r)
        col_pieces = self._get_line_pieces(temp_board, 'col', c)
        
        if len(row_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
            for i in range(4):  # 각 속성에 대해
                if len(set(p[i] for p in row_pieces)) == 1:
                    return True
        
        if len(col_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
            for i in range(4):
                if len(set(p[i] for p in col_pieces)) == 1:
                    return True
        
        # 대각선 체크
        if r == c:  # 주 대각선
            diag_pieces = self._get_line_pieces(temp_board, 'main_diag', 0)
            if len(diag_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
                for i in range(4):
                    if len(set(p[i] for p in diag_pieces)) == 1:
                        return True
        
        if r + c == 3:  # 부 대각선
            diag_pieces = self._get_line_pieces(temp_board, 'anti_diag', 0)
            if len(diag_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
                for i in range(4):
                    if len(set(p[i] for p in diag_pieces)) == 1:
                        return True
        
        # 2x2 블록 체크
        for block_r in range(max(0, r-1), min(3, r+1)):
            for block_c in range(max(0, c-1), min(3, c+1)):
                if block_r+1 < 4 and block_c+1 < 4:
                    block = [
                        temp_board[block_r][block_c],
                        temp_board[block_r][block_c+1],
                        temp_board[block_r+1][block_c],
                        temp_board[block_r+1][block_c+1]
                    ]
                    if block.count(0) == 0:  # 빈칸이 없는 경우만 체크
                        block_pieces = [self.pieces[idx-1] for idx in block]
                        if len(block_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
                            for i in range(4):
                                if len(set(p[i] for p in block_pieces)) == 1:
                                    return True
        
        return False

    def _count_matching_attributes(self, board: List[List[int]], row: int, col: int, piece: Tuple[int,int,int,int]) -> int:
        """
        특정 위치에 피스를 놓았을 때 일치하는 속성의 수를 계산
        """
        # board가 튜플인 경우 리스트로 변환
        board_list = [list(row) for row in board] if isinstance(board, tuple) else board
        matches = 0
        
        # 가로/세로 체크
        row_pieces = self._get_line_pieces(board_list, 'row', row)
        col_pieces = self._get_line_pieces(board_list, 'col', col)
        
        if row_pieces:
            for i in range(4):
                if all(p[i] == piece[i] for p in row_pieces):
                    matches += 1
        
        if col_pieces:
            for i in range(4):
                if all(p[i] == piece[i] for p in col_pieces):
                    matches += 1
        
        # 대각선 체크
        if row == col:
            diag_pieces = self._get_line_pieces(board_list, 'main_diag', 0)
            if diag_pieces:
                for i in range(4):
                    if all(p[i] == piece[i] for p in diag_pieces):
                        matches += 1
        
        if row + col == 3:
            diag_pieces = self._get_line_pieces(board_list, 'anti_diag', 0)
            if diag_pieces:
                for i in range(4):
                    if all(p[i] == piece[i] for p in diag_pieces):
                        matches += 1
        
        return matches

    def _is_immediate_win_for_opponent(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]], piece: Tuple[int,int,int,int]) -> bool:
        """
        해당 피스를 상대방에게 주면 상대방이 즉시 이길 수 있는지 체크
        """
        for r in range(4):
            for c in range(4):
                if board[r][c] == 0:
                    if self._is_winning_move(board, r, c, piece):
                        return True
        return False

    def _is_unavoidable_fork(self, board: List[List[int]], piece: Tuple[int,int,int,int]) -> bool:
        """
        해당 피스를 놓으면 양방 3목 필승이 되는지 체크
        """
        # board가 튜플인 경우 리스트로 변환
        board_list = [list(row) for row in board] if isinstance(board, tuple) else board
        fork_count = 0
        for r in range(4):
            for c in range(4):
                if board_list[r][c] == 0:
                    temp_board = [row.copy() for row in board_list]
                    temp_board[r][c] = self.piece_to_index[piece]
                    if self._count_matching_attributes(temp_board, r, c, piece) >= 3:
                        fork_count += 1
        return fork_count >= 2

    def _has_fork_opportunity(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]], piece: Tuple[int,int,int,int], pos: Tuple[int,int]) -> bool:
        """
        특정 위치에 피스를 놓으면 양방 3목 기회가 생기는지 체크
        """
        # board가 튜플인 경우 리스트로 변환
        board_list = [list(row) for row in board] if isinstance(board, tuple) else board
        temp_board = [row.copy() for row in board_list]
        temp_board[pos[0]][pos[1]] = self.piece_to_index[piece]
        
        three_in_a_row = 0
        
        # 가로/세로 체크
        if self._check_line_win(self._get_line_pieces(temp_board, 'row', pos[0])):
            three_in_a_row += 1
        if self._check_line_win(self._get_line_pieces(temp_board, 'col', pos[1])):
            three_in_a_row += 1
        
        # 대각선 체크
        if pos[0] == pos[1] and self._check_line_win(self._get_line_pieces(temp_board, 'main_diag', 0)):
            three_in_a_row += 1
        if pos[0] + pos[1] == 3 and self._check_line_win(self._get_line_pieces(temp_board, 'anti_diag', 0)):
            three_in_a_row += 1
        
        return three_in_a_row >= 2

    @lru_cache(maxsize=16384)
    def _evaluate_game_state(self, board: Tuple[Tuple[int, ...], ...]) -> float:
        """
        주어진 보드 상태를 P1(최대화 플레이어)의 관점에서 평가합니다.
        점수가 높을수록 P1에게 유리한 상태를 나타냅니다.
        
        Args:
            board: 평가할 게임 보드 상태 (튜플 형태)
        Returns:
            float: 평가 점수
        """

        # 1. 게임 종료 조건 확인 (가장 높은/낮은 우선순위)
        # P1의 승리 (나의 승리)
        if any(self._is_winning_move(list(map(list, board)), r, c, self.index_to_piece[board[r][c]]) 
               for r in range(4) for c in range(4) if board[r][c] != 0 and board[r][c] % 2 != 0):  # P1의 피스
            return self.WIN_SCORE
        
        # P2의 승리 (상대방의 승리)
        if any(self._is_winning_move(list(map(list, board)), r, c, self.index_to_piece[board[r][c]]) 
               for r in range(4) for c in range(4) if board[r][c] != 0 and board[r][c] % 2 == 0):  # P2의 피스
            return self.LOSE_SCORE
        
        # 무승부 (보드가 꽉 찼고 승리 조건 없음)
        if all(cell != 0 for row in board for cell in row):
            return self.DRAW_SCORE

        score = 0.0

        # 2. 중앙 제어 및 코너 제어 (P1의 피스에 가중치)
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        corner_positions = [(0,0), (0,3), (3,0), (3,3)]

        for r, c in center_positions:
            if board[r][c] != 0:
                # P1의 피스 (홀수 인덱스)는 긍정적, P2의 피스 (짝수 인덱스)는 부정적
                score += (1 if board[r][c] % 2 != 0 else -1) * self.CENTER_BONUS
        
        for r, c in corner_positions:
            if board[r][c] != 0:
                score += (1 if board[r][c] % 2 != 0 else -1) * self.CORNER_BONUS

        # 3. 잠재적 라인 평가
        # 각 빈 칸에 대해 P1이 놓을 수 있는 피스들로 평가
        for r, c in self._get_empty_positions(list(map(list, board))):
            # P1이 놓을 수 있는 피스들로 이 위치에 놓았을 때의 잠재력
            for piece in self.available_pieces:
                # 즉시 승리 가능성
                if self._is_winning_move(list(map(list, board)), r, c, piece):
                    score += self.IMMEDIATE_WIN_BONUS
                    continue

                # 3목 기회
                matching_lines = self._count_matching_attributes(list(map(list, board)), r, c, piece)
                if matching_lines >= 1:
                    score += self.THREE_IN_ROW_BONUS * matching_lines

                # 포크 기회
                if self._has_fork_opportunity(list(map(list, board)), self.available_pieces, piece, (r, c)):
                    score += self.FORK_BONUS
        return score