import random
import copy
from typing import List, Tuple
from functools import lru_cache

class P2:
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
        if self._is_unavoidable_fork(piece):
            score += 10
        
        # 2. 3목 가능성 체크 (중간 위험도)
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    if self._count_matching_attributes(r, c, piece) >= 3:
                        score += 5
        
        # 3. 2목 가능성 체크 (낮은 위험도)
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    if self._count_matching_attributes(r, c, piece) >= 2:
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

    def select_piece(self, simulations: int = 2000) -> Tuple[int,int,int,int]:
        """
        상대에게 넘길 피스를 몬테카를로 시뮬레이션으로 평가
        simulations: 각 피스별 시뮬레이션 횟수
        """
        
        print("\n=== 디버깅: select_piece 시작 ===")
        print("현재 보드 상태:")
        for row in self.board:
            print(row)
        print("\n가용한 피스들:", [self._binary_to_mbti(p) for p in self.available_pieces])
        
        # 1. 즉시 승리하는 피스 제외
        safe_pieces = []
        for piece in self.available_pieces:
            if not self._is_immediate_win_for_opponent(piece):
                safe_pieces.append(piece)
            else:
                print(f"⚠️ 즉시 승리 가능한 말 감지: {self._binary_to_mbti(piece)}")
        
        if not safe_pieces:  # 안전한 피스가 없으면 위험도 기반으로 선택
            print("⚠️ 안전한 피스가 없어서 위험도 기반으로 선택")
            # 위험도 점수 계산
            danger_scores = {piece: self._danger_score(piece) for piece in self.available_pieces}
            print("위험도 점수:", {self._binary_to_mbti(p): s for p, s in danger_scores.items()})
            # 가장 위험도가 낮은 피스들 중에서 선택
            min_danger = min(danger_scores.values())
            safe_pieces = [p for p, score in danger_scores.items() if score == min_danger]
            print(f"가장 위험도가 낮은 피스들: {[self._binary_to_mbti(p) for p in safe_pieces]}")
        
        # 2. 공격적인 전략: 상대가 막기 어려운 속성 줄 유도
        attack_scores = {}
        for piece in safe_pieces:
            score = 0
            # 이 피스를 제외한 나머지 피스들로
            remaining_pieces = [p for p in safe_pieces if p != piece]
            
            # 모든 빈 칸에 대해 체크
            for r in range(4):
                for c in range(4):
                    if self.board[r][c] == 0:
                        # 각 피스를 놓아보고 공격 기회가 생기는지 체크
                        for remaining_piece in remaining_pieces:
                            temp_board = [row.copy() for row in self.board]
                            temp_board[r][c] = self.piece_to_index[remaining_piece]
                            
                            # 속성 2개 일치하는 줄이 생기는지 체크
                            if self._count_matching_attributes(r, c, remaining_piece) >= 2:
                                score += 2
                            
                            # 양방 3목 기회가 생기는지 체크
                            if self._has_fork_opportunity(remaining_piece, (r, c)):
                                score += 5  # 양방 3목은 매우 높은 점수
                            
                            # 상대가 막을 수 없는지 체크
                            can_block = False
                            for opp_r in range(4):
                                for opp_c in range(4):
                                    if temp_board[opp_r][opp_c] == 0:
                                        opp_board = [row.copy() for row in temp_board]
                                        opp_board[opp_r][opp_c] = self.piece_to_index[piece]
                                        if not self._is_winning_move(opp_board, opp_r, opp_c, piece):
                                            can_block = True
                                            break
                                if can_block:
                                    break
                            
                            if not can_block:
                                score += 3  # 상대가 막을 수 없는 공격 기회
            
            attack_scores[piece] = score
        
        # 3. 공격 점수가 높은 안전한 피스 선택
        best_score = -1
        best_pieces = []
        for piece in safe_pieces:
            if attack_scores[piece] > best_score:
                best_score = attack_scores[piece]
                best_pieces = [piece]
            elif attack_scores[piece] == best_score:
                best_pieces.append(piece)
        
        selected_piece = random.choice(best_pieces)
        print(f"\n최종 선택된 피스: {self._binary_to_mbti(selected_piece)}")
        print("=== 디버깅: select_piece 종료 ===\n")
        
        return selected_piece

    def place_piece(self, selected_piece: Tuple[int,int,int,int], simulations: int = 1000) -> Tuple[int,int]:
        """
        받은 피스를 놓을 위치를 공격적인 전략으로 평가
        """
        
        idx = self.piece_to_index[selected_piece]
        empty = self._get_empty_positions(self.board)
        
        # 각 위치별 점수 계산
        position_scores = {}
        for r, c in empty:
            score = 0
            temp_board = [row.copy() for row in self.board]
            temp_board[r][c] = idx
            
            # 1. 공격 점수
            # 1-1. 속성 2개 일치하는 줄 만들기
            matches = self._count_matching_attributes(r, c, selected_piece)
            if matches >= 2:
                score += 2
            
            # 1-2. 속성 3개 일치하는 줄 만들기 (거의 승리)
            if self._is_winning_move(temp_board, r, c, selected_piece):
                score += 5
            
            # 1-3. 양방 3목 기회 만들기
            if self._has_fork_opportunity(selected_piece, (r, c)):
                score += 8  # 양방 3목은 매우 높은 점수
            
            # 1-4. 상대가 막을 수 없는 공격 기회
            can_block = False
            for opp_r in range(4):
                for opp_c in range(4):
                    if temp_board[opp_r][opp_c] == 0:
                        for opp_piece in self.available_pieces:
                            if opp_piece != selected_piece:
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
                score += 3
            
            # 2. 방어 점수
            # 2-1. 상대방이 다음 턴에 이길 수 있는지 체크
            for opp_r in range(4):
                for opp_c in range(4):
                    if temp_board[opp_r][opp_c] == 0:
                        for opp_piece in self.available_pieces:
                            if opp_piece != selected_piece:
                                opp_board = [row.copy() for row in temp_board]
                                opp_board[opp_r][opp_c] = self.piece_to_index[opp_piece]
                                if self._is_winning_move(opp_board, opp_r, opp_c, opp_piece):
                                    score -= 10  # 상대가 이길 수 있는 위치는 크게 감점
                                    break
                        if score < 0:
                            break
                if score < 0:
                    break
            
            # 2-2. 상대방이 양방 3목을 만들 수 있는지 체크
            for opp_r in range(4):
                for opp_c in range(4):
                    if temp_board[opp_r][opp_c] == 0:
                        for opp_piece in self.available_pieces:
                            if opp_piece != selected_piece:
                                if self._has_fork_opportunity(opp_piece, (opp_r, opp_c)):
                                    score -= 15  # 양방 3목은 매우 위험하므로 크게 감점
                                    break
                        if score < 0:
                            break
                if score < 0:
                    break
            
            position_scores[(r,c)] = score
        
        # 최고 점수의 위치 선택
        max_score = max(position_scores.values())
        best_positions = [pos for pos, score in position_scores.items() if score == max_score]
        selected_position = random.choice(best_positions)
        
        return selected_position

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
        # 후보 피스를 줬을 때 상대가 승리하는지 시뮬레이션
        board = [row.copy() for row in self.board]
        avail = self.available_pieces.copy()
        avail.remove(piece)
        # 상대가 후보 피스 배치
        r,c = random.choice(self._get_empty_positions(board))
        board[r][c] = self.piece_to_index[piece]
        if self._check_win(board):
            return True
        turn = 0  # 0: 우리 선택(), 1: 상대 선택()
        last_piece = None
        # 남은 피스만큼 무작위 두 플레이어 시뮬레이션
        while avail:
            # 피스 선택
            next_piece = random.choice(avail)
            avail.remove(next_piece)
            # 배치
            r2,c2 = random.choice(self._get_empty_positions(board))
            board[r2][c2] = self.piece_to_index[next_piece]
            if self._check_win(board):
                # 방금 배치한 플레이어가 승리
                return turn == 0  # turn==0이면 상대가 배치 후 승리
            turn ^= 1
        return False

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
        score = 0
        # 중앙 위치 가중치
        if 1 <= pos[0] <= 2 and 1 <= pos[1] <= 2:
            score += 2
        # 모서리 위치 가중치
        if (pos[0] in [0, 3] and pos[1] in [0, 3]):
            score += 1
        # 기존 피스와의 관계 평가
        for r in range(4):
            for c in range(4):
                if board[r][c] != 0:
                    # 같은 속성을 가진 피스 근처에 놓으면 가중치 부여
                    if self.pieces[board[r][c]-1][0] == piece[0]:  # 첫 번째 속성 비교
                        score += 0.5
        return score

    def _evaluate_game_state(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]]) -> float:
        score = 0
        # 가용한 승리 경로 수 계산
        for i in range(4):
            # 가로/세로 라인
            if self._count_winning_paths([board[i][j] for j in range(4)]) > 0:
                score += 1
            if self._count_winning_paths([board[j][i] for j in range(4)]) > 0:
                score += 1
        # 대각선
        if self._count_winning_paths([board[i][i] for i in range(4)]) > 0:
            score += 1
        if self._count_winning_paths([board[i][3-i] for i in range(4)]) > 0:
            score += 1
        # 2x2 블록
        for r in range(3):
            for c in range(3):
                if self._count_winning_paths([board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]) > 0:
                    score += 1
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

    def _count_matching_attributes(self, row: int, col: int, piece: Tuple[int,int,int,int]) -> int:
        """
        특정 위치에 피스를 놓았을 때 일치하는 속성의 수를 계산
        """
        matches = 0
        
        # 가로/세로 체크
        row_pieces = self._get_line_pieces(self.board, 'row', row)
        col_pieces = self._get_line_pieces(self.board, 'col', col)
        
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
            diag_pieces = self._get_line_pieces(self.board, 'main_diag', 0)
            if diag_pieces:
                for i in range(4):
                    if all(p[i] == piece[i] for p in diag_pieces):
                        matches += 1
        
        if row + col == 3:
            diag_pieces = self._get_line_pieces(self.board, 'anti_diag', 0)
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

    def _is_immediate_win_for_opponent(self, piece: Tuple[int,int,int,int]) -> bool:
        """
        해당 피스를 상대방에게 주면 상대방이 즉시 이길 수 있는지 체크
        Returns:
            bool: True면 상대방이 즉시 이길 수 있음, False면 안전함
        """
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:  # 빈 칸인 경우
                    if self._is_winning_move(self.board, r, c, piece):
                        print(f"⚠️ 위험: {self._binary_to_mbti(piece)}를 주면 상대가 ({r},{c})에 놓고 이길 수 있음")
                        return True
        return False

    def _is_unavoidable_fork(self, piece: Tuple[int,int,int,int]) -> bool:
        """
        해당 피스를 놓으면 양방 3목 필승이 되는지 체크
        Returns:
            bool: True면 양방 3목 필승 가능
        """
        fork_count = 0
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    temp_board = [row.copy() for row in self.board]
                    temp_board[r][c] = self.piece_to_index[piece]
                    if self._count_matching_attributes(r, c, piece) >= 3:
                        fork_count += 1
        return fork_count >= 2  # 두 줄 이상이 3목이 됨

    def _has_fork_opportunity(self, piece: Tuple[int,int,int,int], pos: Tuple[int,int]) -> bool:
        """
        특정 위치에 피스를 놓으면 양방 3목 기회가 생기는지 체크
        """
        temp_board = [row.copy() for row in self.board]
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