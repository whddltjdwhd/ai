import random
from typing import List, Tuple, Optional
from functools import lru_cache

class P1:
    def __init__(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]]):
        # 깊은 복사로 안전하게 초기 상태 저장
        self.board = [row.copy() for row in board]
        self.available_pieces = available_pieces.copy()
        # 모든 16개 피스 조합 생성 및 인덱스 매핑
        self.pieces = [(i,j,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.piece_to_index = {p: idx+1 for idx,p in enumerate(self.pieces)}

    def _danger_score(self, piece: Tuple[int,int,int,int]) -> int:
        """
        해당 피스가 얼마나 위험한지 점수 계산
        Returns:
            int: 점수가 높을수록 위험한 피스 (3속성 일치하는 줄이 많을수록 위험)
        """
        score = 0
        # 1. 양방 3목 가능성 체크 (높은 위험도)
        if self._is_unavoidable_fork(self.board, self.available_pieces, piece):
            score += 10
        
        # 2. 3목 가능성 체크 (중간 위험도)
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    if self._count_matching_attributes(self.board, r, c, piece) >= 3:
                        score += 5
        
        # 3. 2목 가능성 체크 (낮은 위험도)
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    if self._count_matching_attributes(self.board, r, c, piece) >= 2:
                        score += 2
        
        return score

    def _binary_to_mbti(self, piece: Tuple[int,int,int,int]) -> str:
        """
        이진수 튜플을 MBTI 형식으로 변환
        Args:
            piece: (0,1)로 이루어진 4개 속성의 튜플
        Returns:
            str: MBTI 형식의 문자열 (예: "INTJ")
        """
        mbti_map = {
            0: ['I', 'N', 'T', 'P'],
            1: ['E', 'S', 'F', 'J']
        }
        return ''.join(mbti_map[bit][i] for i, bit in enumerate(piece))

    def _simulate_loss_rate(self, piece: Tuple[int,int,int,int], trials: int = 100) -> float:
        """
        해당 피스를 줬을 때 상대가 이길 확률을 시뮬레이션으로 계산
        """
        losses = sum(self._simulate_after_give(piece) for _ in range(trials))
        return losses / trials

    def _blocking_complexity(self, piece: Tuple[int,int,int,int]) -> int:
        """
        해당 피스가 상대에게 얼마나 활용하기 어려운지 계산
        값이 낮을수록 상대가 활용하기 어려운 피스
        """
        return sum(self._count_matching_attributes(self.board, r, c, piece)
                  for r in range(4) for c in range(4) if self.board[r][c] == 0)

    def _evaluate_aggression(self, piece: Tuple[int,int,int,int]) -> int:
        """
        해당 피스가 얼마나 공격적인 기회를 만들 수 있는지 평가
        Returns:
            int: 점수가 높을수록 공격적인 기회가 많은 피스
        """
        score = 0
        # 이 피스를 제외한 나머지 피스들로
        remaining_pieces = [p for p in self.available_pieces if p != piece]
        
        # 양방 3목 루프 전략: 연속된 양방 3목 기회 체크
        consecutive_threats = 0
        last_threat_pos = None
        
        # 모든 빈 칸에 대해 체크
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    # 각 피스를 놓아보고 공격 기회가 생기는지 체크
                    for remaining_piece in remaining_pieces:
                        temp_board = [row.copy() for row in self.board]
                        temp_board[r][c] = self.piece_to_index[remaining_piece]
                        
                        # 1. 양방 3목 기회 (가장 높은 점수)
                        if self._has_fork_opportunity(temp_board, self.available_pieces, remaining_piece, (r, c)):
                            score += 10
                            # 연속된 양방 3목 기회 체크
                            if last_threat_pos is not None:
                                if self._can_create_consecutive_threat(temp_board, last_threat_pos, (r, c)):
                                    consecutive_threats += 1
                                    score += 5 * consecutive_threats  # 연속될수록 더 높은 점수
                            last_threat_pos = (r, c)
                        
                        # 2. 속성 3개 일치하는 줄 만들기
                        matches = self._count_matching_attributes(temp_board, r, c, remaining_piece)
                        if matches >= 3:
                            score += 8
                        
                        # 3. 속성 2개 일치하는 줄 만들기
                        elif matches >= 2:
                            score += 3
                        
                        # 4. 상대가 막을 수 없는 공격 기회
                        can_block = False
                        for opp_r in range(4):
                            for opp_c in range(4):
                                if temp_board[opp_r][opp_c] == 0:
                                    for opp_piece in self.available_pieces:
                                        if opp_piece != remaining_piece:
                                            opp_board = [row.copy() for row in temp_board]
                                            opp_board[opp_r][opp_c] = self.piece_to_index[opp_piece]
                                            if not self._is_winning_move(opp_board, opp_r, opp_c, opp_piece):
                                                can_block = True
                                                break
                                    if can_block:
                                        break
                            if can_block:
                                break
                        
                        if not can_block:
                            score += 5
        
        # 희생 피스 유도 전략: 남은 피스가 적을 때 강제 패배 피스 체크
        if len(self.available_pieces) <= 4:  # 남은 피스가 4개 이하일 때
            forced_lose_pieces = self._find_forced_lose_pieces(self.board, self.available_pieces, piece)
            if forced_lose_pieces:
                score += 15  # 강제 패배 피스가 있으면 매우 높은 점수
        
        return score

    def _can_create_consecutive_threat(self, board: List[List[int]], pos1: Tuple[int,int], pos2: Tuple[int,int]) -> bool:
        """
        두 위치가 연속된 위협을 만들 수 있는지 체크
        """
        # 두 위치가 같은 행/열/대각선에 있는지 체크
        if pos1[0] == pos2[0] or pos1[1] == pos2[1] or \
           abs(pos1[0] - pos2[0]) == abs(pos1[1] - pos2[1]):
            return True
        return False

    def _find_forced_lose_pieces(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]], piece: Tuple[int,int,int,int]) -> List[Tuple[int,int,int,int]]:
        """
        해당 피스를 줬을 때 상대가 반드시 지는 피스들을 찾음
        """
        forced_lose = []
        remaining_pieces = [p for p in available_pieces if p != piece]
        
        for test_piece in remaining_pieces:
            # 이 피스를 줬을 때 상대가 이길 수 있는 방법이 없는지 체크
            can_win = False
            for r in range(4):
                for c in range(4):
                    if board[r][c] == 0:
                        temp_board = [row.copy() for row in board]
                        temp_board[r][c] = self.piece_to_index[test_piece]
                        if self._is_winning_move(temp_board, r, c, test_piece):
                            can_win = True
                            break
                if can_win:
                    break
            
            if not can_win:
                forced_lose.append(test_piece)
        
        return forced_lose

    def select_piece(self, simulations: int = 2000) -> Tuple[int,int,int,int]:
        """
        더 최적화된 피스 선택 로직
        """
        print("\n===== [P1] 피스 선택 단계 =====")
        # 1. 즉시 승리하는 피스 제외
        safe_pieces = [p for p in self.available_pieces if not self._is_immediate_win_for_opponent(self.board, self.available_pieces, p)]
        
        if not safe_pieces:
            danger_scores = {p: self._danger_score(p) for p in self.available_pieces}
            min_danger = min(danger_scores.values())
            safe_pieces = [p for p, score in danger_scores.items() if score == min_danger]
            return random.choice(safe_pieces)
        
        # 2. 동적 깊이 조정 (더 얕게)
        depth = 2 if len(safe_pieces) <= 4 else 1
        
        # 3. Minimax로 각 피스 평가
        best_score = float('-inf')
        best_pieces = []
        
        for piece in safe_pieces:
            score, _, _ = self.minimax(self.board, self.available_pieces, depth, float('-inf'), float('inf'), False)
            
            if score > best_score:
                best_score = score
                best_pieces = [piece]
            elif score == best_score:
                best_pieces.append(piece)
        
        return random.choice(best_pieces)

    def place_piece(self, selected_piece: Tuple[int,int,int,int], simulations: int = 1000) -> Tuple[int,int]:
        """
        더 최적화된 위치 선택 로직
        """
        print("\n===== [P1] 피스 배치 단계 =====")
        # 1. 승리 가능한 위치 먼저 체크
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    if self._is_winning_move(self.board, r, c, selected_piece):
                        return (r, c)
        
        # 2. 양방 3목 기회 체크
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    if self._has_fork_opportunity(self.board, self.available_pieces, selected_piece, (r, c)):
                        return (r, c)
        
        # 3. Minimax로 최적의 위치 찾기 (깊이 1로 제한)
        remaining_pieces = [p for p in self.available_pieces if p != selected_piece]
        _, _, best_pos = self.minimax(self.board, remaining_pieces, depth=1, alpha=float('-inf'), beta=float('inf'), is_maximizing=True)
        
        # 4. Minimax가 실패하면 휴리스틱으로 폴백
        if best_pos is None:
            # 4-1. 중앙 위치 우선
            if self.board[1][1] == 0:
                return (1, 1)
            if self.board[1][2] == 0:
                return (1, 2)
            if self.board[2][1] == 0:
                return (2, 1)
            if self.board[2][2] == 0:
                return (2, 2)
            
            # 4-2. 모서리 위치
            corners = [(0,0), (0,3), (3,0), (3,3)]
            for r, c in corners:
                if self.board[r][c] == 0:
                    return (r, c)
            
            # 4-3. 첫 번째 빈 칸
            for r in range(4):
                for c in range(4):
                    if self.board[r][c] == 0:
                        return (r, c)
        
        return best_pos

    def _place_piece_heuristic(self, selected_piece: Tuple[int,int,int,int]) -> Tuple[int,int]:
        """
        기존의 휴리스틱 기반 위치 선택 로직
        """
        # 기존 place_piece 메서드의 로직을 여기로 이동
        # ... (기존 코드 유지)

    def _can_win_with_pieces(self, pieces: List[Tuple[int,int,int,int]]) -> bool:
        """
        주어진 피스들로 우리가 이길 수 있는지 확인
        """
        # 모든 빈 칸에 대해 체크
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:  # 빈 칸인 경우
                    # 각 피스를 놓아보고 이길 수 있는지 체크
                    for piece in pieces:
                        if self._is_winning_move(self.board, r, c, piece):
                            return True
        return False

    def _simulate_after_give(self, piece: Tuple[int,int,int,int]) -> bool:
        """
        후보 피스를 줬을 때 우리가 이기는지 시뮬레이션
        Returns:
            bool: True면 우리가 이김, False면 상대가 이김
        """
        # 후보 피스를 줬을 때 우리가 이기는지 시뮬레이션
        board = [row.copy() for row in self.board]
        avail = self.available_pieces.copy()
        avail.remove(piece)
        
        # 상대가 후보 피스 배치
        r,c = random.choice(self._get_empty_positions(board))
        board[r][c] = self.piece_to_index[piece]
        if self._check_win(board):
            return False  # 상대가 이김
        
        turn = 0  # 0: 우리 선택, 1: 상대 선택
        while avail:
            # 피스 선택
            next_piece = random.choice(avail)
            avail.remove(next_piece)
            
            # 배치
            r2,c2 = random.choice(self._get_empty_positions(board))
            board[r2][c2] = self.piece_to_index[next_piece]
            
            if self._check_win(board):
                return turn == 0  # turn==0이면 우리가 이김
            
            turn ^= 1
        
        return False  # 무승부는 우리가 진 것으로 간주

    def _simulate_win_rate(self, piece: Tuple[int,int,int,int], trials: int = 5000) -> float:
        """
        해당 피스를 줬을 때 우리가 이길 확률을 시뮬레이션으로 계산
        Args:
            piece: 시뮬레이션할 피스
            trials: 시뮬레이션 횟수 (기본값: 5000)
        Returns:
            float: 승률 (0.0 ~ 1.0)
        """
        print(f"\n{self._binary_to_mbti(piece)} 시뮬레이션 시작...")
        wins = 0
        for i in range(trials):
            if not self._simulate_after_give(piece):
                wins += 1
            if (i + 1) % 500 == 0:
                print(f"진행률: {(i + 1) / trials * 100:.1f}% (현재 승률: {wins / (i + 1) * 100:.1f}%)")
        win_rate = wins / trials
        print(f"{self._binary_to_mbti(piece)} 최종 승률: {win_rate * 100:.1f}%")
        return win_rate

    def _simulate_after_place(self, piece: Tuple[int,int,int,int], pos: Tuple[int,int]) -> bool:
        board = [row.copy() for row in self.board]
        avail = self.available_pieces.copy()
        avail.remove(piece)
        board[pos[0]][pos[1]] = self.piece_to_index[piece]
        
        # 즉시 승리 체크
        if self._check_win(board):
            return True
        
        # 상대방이 즉시 승리할 수 있는 위치 체크
        for r,c in self._get_empty_positions(board):
            for next_piece in avail:
                temp = [row.copy() for row in board]
                temp[r][c] = self.piece_to_index[next_piece]
                if self._check_win(temp):
                    return False  # 상대방이 이길 수 있는 위치는 피함
                
        # 나머지 시뮬레이션 진행
        turn = 1
        while avail:
            nxt = random.choice(avail)
            avail.remove(nxt)
            r,c = random.choice(self._get_empty_positions(board))
            board[r][c] = self.piece_to_index[nxt]
            if self._check_win(board):
                return turn == 0
            turn ^= 1
        return False

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
        if line.count(0) > 1: return False  # 빈 칸이 2개 이상이면 체크할 필요 없음
        attrs = [self.pieces[idx-1] for idx in line if idx != 0]  # 빈 칸 제외
        if len(attrs) < 3: return False  # 3개 미만이면 체크할 필요 없음
        
        for i in range(4):  # 각 속성에 대해
            matching_count = sum(1 for a in attrs if a[i] == attrs[0][i])
            if matching_count >= 3:  # 3개 이상 일치하면 승리
                return True
        return False

    def _check_2x2(self, block: List[int]) -> bool:
        if block.count(0) > 1: return False  # 빈 칸이 2개 이상이면 체크할 필요 없음
        attrs = [self.pieces[idx-1] for idx in block if idx != 0]  # 빈 칸 제외
        if len(attrs) < 3: return False  # 3개 미만이면 체크할 필요 없음
        
        for i in range(4):  # 각 속성에 대해
            matching_count = sum(1 for a in attrs if a[i] == attrs[0][i])
            if matching_count >= 3:  # 3개 이상 일치하면 승리
                return True
        return False

    def _evaluate_position(self, board: List[List[int]], pos: Tuple[int, int], piece: Tuple[int,int,int,int]) -> float:
        """
        개선된 위치 평가 함수
        """
        score = 0
        
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
        
        return score

    @lru_cache(maxsize=1024)
    def _evaluate_game_state(self, board_tuple: Tuple[Tuple[int, ...], ...], pieces_tuple: Tuple[Tuple[int,int,int,int], ...], is_maximizing: bool = True) -> float:
        """
        개선된 게임 상태 평가 함수
        """
        board = [list(row) for row in board_tuple]
        current_available_pieces = list(pieces_tuple)
        
        # 1. 승리 조건 체크
        if self._check_win(board):
            return float('inf') if is_maximizing else float('-inf')
        
        score = 0
        
        # 2. 중앙 제어 점수 추가
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        for r, c in center_positions:
            if board[r][c] != 0:
                piece = self.pieces[board[r][c]-1]
                # 내 피스가 중앙에 있으면 +5점, 상대 피스가 중앙에 있으면 -3점
                if is_maximizing:
                    if piece in current_available_pieces:
                        score += 5
                    else:
                        score -= 3
                else:
                    if piece in current_available_pieces:
                        score -= 5
                    else:
                        score += 3
        
        # 3. 양방 3목 기회 체크
        for r in range(4):
            for c in range(4):
                if board[r][c] == 0:
                    for piece in current_available_pieces:
                        if self._has_fork_opportunity(board, current_available_pieces, piece, (r, c)):
                            return 1000 if is_maximizing else -1000
        
        # 4. 3목 기회 체크
        for i in range(4):
            # 가로/세로 라인
            row_pieces = [self.pieces[board[i][j]-1] for j in range(4) if board[i][j] != 0]
            col_pieces = [self.pieces[board[j][i]-1] for j in range(4) if board[j][i] != 0]
            
            for pieces in [row_pieces, col_pieces]:
                if len(pieces) >= 2:
                    for attr in range(4):
                        matches = sum(1 for p in pieces if p[attr] == pieces[0][attr])
                        if matches == 3:
                            score += 100 if is_maximizing else -100
        
        # 5. 피스 활용도 평가 추가
        if len(current_available_pieces) <= 4:  # 게임 후반
            # 5-1. 강제 패배 피스 체크
            forced_lose_pieces = self._find_forced_lose_pieces(board, current_available_pieces, current_available_pieces[0])
            if forced_lose_pieces:
                score += 40 if is_maximizing else -40
            
            # 5-2. 남은 피스들의 속성 분포 분석
            attribute_counts = {i: 0 for i in range(4)}  # 각 속성별 남은 피스 수
            for piece in current_available_pieces:
                for i, attr in enumerate(piece):
                    attribute_counts[i] += attr
            
            # 5-3. 균형 잡힌 속성 분포에 가중치 부여
            balanced_score = sum(1 for count in attribute_counts.values() if 1 <= count <= 2)
            score += balanced_score * 10 if is_maximizing else -balanced_score * 10
        
        # 6. 2x2 블록 평가
        for r in range(3):
            for c in range(3):
                block = [
                    board[r][c], board[r][c+1],
                    board[r+1][c], board[r+1][c+1]
                ]
                if block.count(0) <= 1:  # 빈 칸이 1개 이하
                    block_pieces = [self.pieces[idx-1] for idx in block if idx != 0]
                    if len(block_pieces) >= 3:
                        for attr in range(4):
                            matches = sum(1 for p in block_pieces if p[attr] == block_pieces[0][attr])
                            if matches == 3:
                                score += 50 if is_maximizing else -50
        
        return score

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
        temp_board = [row.copy() for row in board]
        temp_board[r][c] = piece_idx
        
        # 가로/세로 체크
        row_pieces = self._get_line_pieces(temp_board, 'row', r)
        col_pieces = self._get_line_pieces(temp_board, 'col', c)
        
        if len(row_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
            for i in range(4):  # 각 속성에 대해
                if len(set(p[i] for p in row_pieces)) == 1:
                    print(f"가로 승리: ({r},{c})에 {self._binary_to_mbti(piece)} 놓으면 승리")
                    return True
        
        if len(col_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
            for i in range(4):
                if len(set(p[i] for p in col_pieces)) == 1:
                    print(f"세로 승리: ({r},{c})에 {self._binary_to_mbti(piece)} 놓으면 승리")
                    return True
        
        # 대각선 체크
        if r == c:  # 주 대각선
            diag_pieces = self._get_line_pieces(temp_board, 'main_diag', 0)
            if len(diag_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
                for i in range(4):
                    if len(set(p[i] for p in diag_pieces)) == 1:
                        print(f"주대각선 승리: ({r},{c})에 {self._binary_to_mbti(piece)} 놓으면 승리")
                        return True
        
        if r + c == 3:  # 부 대각선
            diag_pieces = self._get_line_pieces(temp_board, 'anti_diag', 0)
            if len(diag_pieces) >= 4:  # 4개 이상의 말이 있는 경우만 체크
                for i in range(4):
                    if len(set(p[i] for p in diag_pieces)) == 1:
                        print(f"부대각선 승리: ({r},{c})에 {self._binary_to_mbti(piece)} 놓으면 승리")
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
                                    print(f"2x2블록 승리: ({r},{c})에 {self._binary_to_mbti(piece)} 놓으면 승리")
                                    return True
        
        return False

    def _check_opponent_winning_move(self, piece: Tuple[int,int,int,int]) -> bool:
        """
        상대방이 이 피스를 받았을 때 무조건 이길 수 있는지 확인
        """
        # 모든 빈 칸에 대해 체크
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:  # 빈 칸인 경우
                    if self._is_winning_move(self.board, r, c, piece):
                        return True  # 이 피스를 주면 상대방이 이길 수 있음
        
        return False  # 이 피스를 줘도 상대방이 이길 수 없음

    def _check_opponent_next_win(self, board: List[List[int]]) -> bool:
        """
        상대방이 다음 턴에 승리할 수 있는지 확인
        """
        for piece in self.available_pieces:
            for r in range(4):
                for c in range(4):
                    if board[r][c] == 0:
                        temp = [row.copy() for row in board]
                        temp[r][c] = self.piece_to_index[piece]
                        if self._check_win(temp):
                            return True
        return False

    def _count_matching_attributes(self, board: List[List[int]], row: int, col: int, piece: Tuple[int,int,int,int]) -> int:
        """
        특정 위치에 피스를 놓았을 때 일치하는 속성의 수를 계산
        """
        matches = 0
        
        # 가로/세로 체크
        row_pieces = self._get_line_pieces(board, 'row', row)
        col_pieces = self._get_line_pieces(board, 'col', col)
        
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
            diag_pieces = self._get_line_pieces(board, 'main_diag', 0)
            if diag_pieces:
                for i in range(4):
                    if all(p[i] == piece[i] for p in diag_pieces):
                        matches += 1
        
        if row + col == 3:
            diag_pieces = self._get_line_pieces(board, 'anti_diag', 0)
            if diag_pieces:
                for i in range(4):
                    if all(p[i] == piece[i] for p in diag_pieces):
                        matches += 1
        
        return matches

    def _is_safe_piece(self, piece: Tuple[int,int,int,int]) -> bool:
        """
        해당 피스가 상대방에게 안전한지 체크
        """
        # 모든 빈 칸에 대해 체크
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    temp_board = [row.copy() for row in self.board]
                    temp_board[r][c] = self.piece_to_index[piece]
                    
                    # 1. 즉시 승리 체크
                    if self._is_winning_move(temp_board, r, c, piece):
                        return False
                    
                    # 2. 다음 턴에 우리가 막을 수 없는 승리 수 체크
                    can_block = False
                    for next_piece in self.available_pieces:
                        if next_piece != piece:
                            for next_r in range(4):
                                for next_c in range(4):
                                    if temp_board[next_r][next_c] == 0:
                                        next_board = [row.copy() for row in temp_board]
                                        next_board[next_r][next_c] = self.piece_to_index[next_piece]
                                        # 우리가 놓은 후에도 상대방이 이길 수 있는지 체크
                                        opponent_can_win = False
                                        for opp_r in range(4):
                                            for opp_c in range(4):
                                                if next_board[opp_r][opp_c] == 0:
                                                    final_board = [row.copy() for row in next_board]
                                                    final_board[opp_r][opp_c] = self.piece_to_index[piece]
                                                    if self._is_winning_move(final_board, opp_r, opp_c, piece):
                                                        opponent_can_win = True
                                                        break
                                            if opponent_can_win:
                                                break
                                        if not opponent_can_win:
                                            can_block = True
                                            break
                                if can_block:
                                    break
                            if can_block:
                                break
                    
                    if not can_block:
                        return False
        return True

    def _is_immediate_win_for_opponent(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]], piece: Tuple[int,int,int,int]) -> bool:
        """
        해당 피스를 상대방에게 주면 상대방이 즉시 이길 수 있는지 체크
        """
        for r in range(4):
            for c in range(4):
                if board[r][c] == 0:
                    if self._is_winning_move(board, r, c, piece):
                        print(f"⚠️ 위험: {self._binary_to_mbti(piece)}를 주면 상대가 ({r},{c})에 놓고 이길 수 있음")
                        return True
        return False

    def _is_unavoidable_fork(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]], piece: Tuple[int,int,int,int]) -> bool:
        """
        해당 피스를 놓으면 양방 3목 필승이 되는지 체크
        """
        fork_count = 0
        for r in range(4):
            for c in range(4):
                if board[r][c] == 0:
                    temp_board = [row.copy() for row in board]
                    temp_board[r][c] = self.piece_to_index[piece]
                    if self._count_matching_attributes(temp_board, r, c, piece) >= 3:
                        fork_count += 1
        return fork_count >= 2

    def _has_fork_opportunity(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]], piece: Tuple[int,int,int,int], pos: Tuple[int,int]) -> bool:
        """
        특정 위치에 피스를 놓으면 양방 3목 기회가 생기는지 체크
        """
        temp_board = [row.copy() for row in board]
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

    def minimax(self, board: List[List[int]], current_available_pieces: List[Tuple[int,int,int,int]], depth: int, alpha: float, beta: float, is_maximizing: bool) -> Tuple[float, Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int]]]:
        """
        더 최적화된 Minimax 알고리즘
        """
        # 1. 초기 가지치기: 승리 조건 체크
        if self._check_win(board):
            return (float('inf') if is_maximizing else float('-inf')), None, None
        
        # 2. 종료 조건 체크
        if depth == 0 or not current_available_pieces:
            board_tuple = tuple(tuple(row) for row in board)
            pieces_tuple = tuple(current_available_pieces)
            return self._evaluate_game_state(board_tuple, pieces_tuple, is_maximizing), None, None
        
        # 3. 이동 순서 최적화를 위한 후보 생성
        moves = self._get_ordered_moves(board, current_available_pieces, is_maximizing)
        
        if is_maximizing:
            max_eval = float('-inf')
            best_piece = None
            best_pos = None
            
            for move in moves:
                r, c, piece = move
                temp_board = [row.copy() for row in board]
                temp_board[r][c] = self.piece_to_index[piece]
                remaining_pieces = [p for p in current_available_pieces if p != piece]
                
                # 4. 빠른 승리 체크
                if self._is_winning_move(temp_board, r, c, piece):
                    return float('inf'), piece, (r, c)
                
                eval_score, _, _ = self.minimax(temp_board, remaining_pieces, depth-1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_piece = piece
                    best_pos = (r, c)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_piece, best_pos
        
        else:
            min_eval = float('inf')
            best_piece = None
            best_pos = None
            
            for move in moves:
                r, c, piece = move
                temp_board = [row.copy() for row in board]
                temp_board[r][c] = self.piece_to_index[piece]
                remaining_pieces = [p for p in current_available_pieces if p != piece]
                
                # 4. 빠른 패배 체크
                if self._is_winning_move(temp_board, r, c, piece):
                    return float('-inf'), piece, (r, c)
                
                eval_score, _, _ = self.minimax(temp_board, remaining_pieces, depth-1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_piece = piece
                    best_pos = (r, c)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_piece, best_pos

    def _get_ordered_moves(self, board: List[List[int]], current_available_pieces: List[Tuple[int,int,int,int]], is_maximizing: bool) -> List[Tuple[int, int, Tuple[int,int,int,int]]]:
        """
        이동 순서를 최적화하여 반환
        """
        moves = []
        empty_positions = self._get_empty_positions(board)
        
        # 1. 승리 가능한 수를 먼저 평가
        for r, c in empty_positions:
            for piece in current_available_pieces:
                if self._is_winning_move(board, r, c, piece):
                    return [(r, c, piece)]  # 승리 수가 있으면 즉시 반환
        
        # 2. 양방 3목 기회가 있는 수를 다음으로 평가
        for r, c in empty_positions:
            for piece in current_available_pieces:
                if self._has_fork_opportunity(board, current_available_pieces, piece, (r, c)):
                    moves.append((r, c, piece))
        
        # 3. 나머지 수들을 평가 점수 순으로 정렬
        remaining_moves = []
        for r, c in empty_positions:
            for piece in current_available_pieces:
                if (r, c, piece) not in moves:
                    score = self._evaluate_position(board, (r, c), piece)
                    remaining_moves.append((score, r, c, piece))
        
        # 평가 점수 순으로 정렬 (최대화 플레이어는 높은 점수부터, 최소화 플레이어는 낮은 점수부터)
        remaining_moves.sort(reverse=is_maximizing)
        moves.extend([(r, c, piece) for _, r, c, piece in remaining_moves])
        
        return moves