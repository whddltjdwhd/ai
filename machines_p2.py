import random
from typing import List, Tuple, Optional
from functools import lru_cache
import sys

class P2:
    # 게임 평가 상수 (P2 관점)
    WIN_SCORE = 10000.0
    LOSE_SCORE = -10000.0
    DRAW_SCORE = 0.0
    FORK_BONUS = 500.0  # 양방 3목 기회
    MATCHING_ATTRIBUTES_BONUS = 10.0  # 3개 속성 일치 라인 하나당 보너스
    CENTER_BONUS = 20.0  # 중앙 위치 보너스
    CORNER_BONUS = 10.0  # 코너 위치 보너스
    IMMEDIATE_WIN_BONUS = 1000.0  # 즉시 승리 기회
    THREE_IN_ROW_BONUS = 100.0  # 3목 기회

    def __init__(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]]):
        # 깊은 복사로 안전하게 초기 상태 저장
        self.board = [row.copy() for row in board]
        self.available_pieces = available_pieces.copy()
        # 모든 16개 피스 조합 생성 및 인덱스 매핑
        self.pieces = [(i,j,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.piece_to_index = {p: idx+1 for idx,p in enumerate(self.pieces)}
        self.index_to_piece = {idx+1: p for idx,p in enumerate(self.pieces)}  # 인덱스로 피스 찾기 추가
        self.minimax_depth = self._get_minimax_depth()  # 동적 깊이 설정
        self.chosen_piece = None  # place_piece에서 결정된 '상대에게 줄 피스'를 저장할 변수
        self.debug = True

    def _get_minimax_depth(self) -> int:
        """
        게임 상태에 따라 적절한 Minimax 깊이를 반환
        Returns:
            int: Minimax 탐색 깊이
        """
        empty_count = sum(1 for row in self.board for cell in row if cell == 0)
        
        # 게임 초반 (12개 이상 빈칸)
        if empty_count >= 12:
            return 2  # 초반에도 적당한 깊이로 탐색
        
        # 게임 중반 (8-11개 빈칸)
        elif empty_count >= 8:
            return 3  # 중반에는 좀 더 깊이 탐색
        
        # 게임 후반 (4-7개 빈칸)
        elif empty_count >= 4:
            return 4  # 후반에는 더 깊이 탐색
        
        # 게임 막바지 (3개 이하 빈칸)
        else:
            return 5  # 막바지에는 최대한 깊게 탐색

    def _danger_score(self, board: Tuple[Tuple[int, ...], ...], current_available_pieces: Tuple[Tuple[int,int,int,int], ...], piece: Tuple[int,int,int,int]) -> float:
        """
        주어진 피스가 상대방(P1)에게 얼마나 위험한지 점수를 계산합니다.
        점수가 높을수록 P2에게는 위험한 피스 (즉, 상대에게는 좋은 피스)입니다.
        이 함수는 P2가 P1에게 줄 피스를 선택할 때 활용됩니다.
        
        Args:
            board: 현재 게임 보드 상태 (튜플 형태)
            current_available_pieces: P2가 현재 가진 피스 목록 (튜플 형태)
            piece: 평가할 피스
        Returns:
            float: 위험도 점수
        """
        if self.debug:
            sys.stdout.write(f"  [Danger] 피스 {self._binary_to_mbti(piece)}의 위험도 평가 시작...\n")
            sys.stdout.flush()

        score = 0.0
        
        # 1. 즉시 승리 가능성 체크 (매우 높은 위험도)
        for r, c in self._get_empty_positions(board):
            if self._is_winning_move(board, r, c, piece):
                if self.debug:
                    sys.stdout.write(f"  [Danger] 즉시 승리 가능성 감지! Score: {self.IMMEDIATE_WIN_BONUS}\n")
                    sys.stdout.flush()
                return self.IMMEDIATE_WIN_BONUS

        # 2. 양방 3목 가능성 체크 (높은 위험도)
        if self._is_unavoidable_fork(board, piece):
            if self.debug:
                sys.stdout.write(f"  [Danger] 양방 3목 가능성 감지! Score: {self.FORK_BONUS}\n")
                sys.stdout.flush()
            score += self.FORK_BONUS

        # 3. 3목 가능성 체크 (중간 위험도)
        for r, c in self._get_empty_positions(board):
            matching_lines = self._count_matching_attributes(board, r, c, piece)
            if matching_lines >= 1:
                score += self.THREE_IN_ROW_BONUS * matching_lines
                if self.debug:
                    sys.stdout.write(f"  [Danger] 위치 ({r},{c})에서 {matching_lines}개의 3목 가능성 감지! 현재 점수: {score}\n")
                    sys.stdout.flush()

        # 4. 중앙/코너 제어 가능성 체크
        for r, c in [(1,1), (1,2), (2,1), (2,2)]:  # 중앙 위치
            if board[r][c] == 0:  # 빈 칸인 경우
                score += self.CENTER_BONUS * 0.5  # 중앙 제어 가능성에 대한 보너스
        for r, c in [(0,0), (0,3), (3,0), (3,3)]:  # 코너 위치
            if board[r][c] == 0:  # 빈 칸인 경우
                score += self.CORNER_BONUS * 0.5  # 코너 제어 가능성에 대한 보너스

        # 5. 상대방의 남은 피스들과의 관계 평가
        # 현재 피스와 상대방의 남은 피스들 간의 보완 관계를 평가
        for other_piece in current_available_pieces:
            if other_piece != piece:
                # 두 피스가 3개 이상의 속성을 공유하는지 확인
                matching_attrs = sum(1 for i in range(4) if piece[i] == other_piece[i])
                if matching_attrs >= 3:
                    score += self.MATCHING_ATTRIBUTES_BONUS * matching_attrs

        if self.debug:
            sys.stdout.write(f"  [Danger] 최종 위험도 점수: {score}\n")
            sys.stdout.flush()
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

    def place_piece(self, selected_piece: Tuple[int,int,int,int]) -> Tuple[int,int]:
        """
        P2가 Minimax를 사용하여 최적의 위치와 다음 턴에 상대에게 줄 피스를 결정합니다.
        P2는 항상 후공이므로, Minimax 탐색 시 최소화 플레이어의 관점에서 시작합니다.
        Args:
            selected_piece: P1이 P2에게 준, P2가 보드에 놓을 피스
        Returns:
            Tuple[int, int]: 피스를 놓을 위치 (r, c)
        """
        print("\n===== [P2] 피스 배치 단계 =====")
        print(f"depth: {self.minimax_depth}")
        print(f"배치할 피스 (P1이 준 피스): {self._binary_to_mbti(selected_piece)}")
        
        self.minimax_depth = self._get_minimax_depth()  # 매 턴마다 깊이 업데이트
        print(f"현재 Minimax 깊이: {self.minimax_depth}")
        
        empty_count = sum(1 for row in self.board for cell in row if cell == 0)
        print(f"보드 상태: {16-empty_count}/16칸 배치됨")
        print(f"남은 피스: {len(self.available_pieces)}")
        
        # Minimax 호출 전에 캐시 초기화
        self.minimax.cache_clear()
        self._evaluate_game_state.cache_clear()
        
        # Minimax 호출 (P2는 항상 후공이므로 is_maximizing_player=False)
        eval_score, best_piece_to_give, best_pos_to_place = self.minimax(
            tuple(tuple(r) for r in self.board),  # 현재 보드 상태 (튜플)
            selected_piece,  # P2가 놓을 피스 (P1이 P2에게 준 피스)
            tuple(self.available_pieces),  # P2가 P1에게 줄 수 있는 피스 목록
            self.minimax_depth,  # 동적으로 조정된 깊이 사용
            float('-inf'), 
            float('inf'),
            False  # P2는 최소화 플레이어
        )
        
        # Minimax가 결정한 최적의 '상대에게 줄 피스'를 저장
        if best_pos_to_place is None or best_piece_to_give is None:
            print("⚠️ Warning: Minimax가 최적의 위치 또는 피스를 선택하지 못했습니다. 대체 로직 사용.")
            
            # --- 대체 로직: 놓을 위치 선택 ---
            empty_positions = self._get_empty_positions(self.board)
            if not empty_positions:
                print("⚠️ Error: 더 이상 놓을 곳이 없습니다. (게임 종료 예상)")
                return (0, 0)
            
            position_scores = []
            for r, c in empty_positions:
                # P2의 관점에서 해당 위치에 selected_piece를 놓았을 때의 점수를 평가
                score_at_pos = self._evaluate_position(tuple(tuple(row) for row in self.board), (r, c), selected_piece)
                position_scores.append((score_at_pos, (r, c)))
            
            # P2는 자신의 점수를 최대화해야 하므로 내림차순 정렬
            position_scores.sort(key=lambda x: x[0], reverse=True)
            best_pos_to_place = position_scores[0][1]
            print(f"  - 대체 선택 위치: {best_pos_to_place} (상위 3개 후보):")
            for i, (score, pos) in enumerate(position_scores[:3]):
                print(f"    {i+1}. 위치: {pos}, 점수: {score:.2f}")

            # --- 대체 로직: 상대에게 줄 피스 선택 ---
            # P2가 놓은 selected_piece를 제외한 남은 피스들
            remaining_pieces_for_giving_to_p1 = list(self.available_pieces)
            if selected_piece in remaining_pieces_for_giving_to_p1:
                remaining_pieces_for_giving_to_p1.remove(selected_piece)
            
            if not remaining_pieces_for_giving_to_p1:
                print("⚠️ Error: 상대에게 줄 피스가 남아있지 않습니다. (이전 턴에서 모든 피스 소진 예상)")
                self.chosen_piece = (0,0,0,0)
            else:
                # _danger_score를 사용하여 위험도가 가장 낮은 피스 선택
                piece_danger_scores = []
                for piece_candidate in remaining_pieces_for_giving_to_p1:
                    danger = self._danger_score(
                        tuple(tuple(r) for r in self.board),  # 현재 보드
                        tuple(p for p in remaining_pieces_for_giving_to_p1 if p != piece_candidate),  # piece_candidate를 제외한 나머지 피스들
                        piece_candidate  # P1에게 줄 피스 후보
                    )
                    piece_danger_scores.append((danger, piece_candidate))
                
                # 위험도가 낮은 순으로 정렬 (오름차순)
                piece_danger_scores.sort(key=lambda x: x[0])
                self.chosen_piece = piece_danger_scores[0][1]  # 가장 위험도가 낮은 피스 선택
                print(f"  - 대체 선택 피스: {self._binary_to_mbti(self.chosen_piece)} (상위 3개 후보):")
                for i, (danger_score, piece) in enumerate(piece_danger_scores[:3]):
                    print(f"    {i+1}. 피스: {self._binary_to_mbti(piece)}, 위험도: {danger_score:.2f}")

        else:  # Minimax가 성공적으로 최적의 수를 찾은 경우
            self.chosen_piece = best_piece_to_give
            print(f"Minimax 선택 위치: {best_pos_to_place} (평가 점수: {eval_score:.2f})")
            print(f"다음 턴에 줄 피스: {self._binary_to_mbti(self.chosen_piece)}")
        
        return best_pos_to_place

    def select_piece(self) -> Tuple[int,int,int,int]:
        """
        place_piece 단계에서 Minimax가 결정한 '상대에게 줄 최적의 피스'를 반환
        Returns:
            Tuple[int,int,int,int]: 상대에게 줄 최적의 피스
        """
        print("\n===== [P1] 피스 선택 단계 =====")
        print(f"depth: {self.minimax_depth}")
        print(f"남은 피스: {len(self.available_pieces)}")
        
        if self.chosen_piece is None:
            # 비상 상황: place_piece가 호출되지 않았거나, Minimax가 결과를 내지 못한 경우
            print("🚨 Warning: chosen_piece가 설정되지 않았습니다. 대체 로직 사용.")
            
            # 기존 _danger_score 로직으로 가장 덜 위험한 피스 선택
            safe_pieces = []
            for piece_candidate in self.available_pieces:
                if not self._is_immediate_win_for_opponent(self.board, self.available_pieces, piece_candidate):
                    safe_pieces.append(piece_candidate)
            
            if safe_pieces:
                # 위험도 점수 계산 및 정렬
                piece_scores = [(p, self._danger_score(self.board, self.available_pieces, p)) for p in safe_pieces]
                piece_scores.sort(key=lambda x: x[1])
                
                print("\n안전한 피스 목록 (위험도 순):")
                for piece, score in piece_scores[:3]:  # 상위 3개만 출력
                    print(f"  {self._binary_to_mbti(piece)}: {score:.2f}")
                
                selected = piece_scores[0][0]
                print(f"\n선택된 피스: {self._binary_to_mbti(selected)} (위험도: {piece_scores[0][1]:.2f})")
                self.chosen_piece = selected  # 선택된 피스를 chosen_piece에 저장
                return selected
            else:
                # 모든 피스가 상대에게 즉시 승리 기회를 주는 경우 (최후의 선택)
                print("\n⚠️ 모든 피스가 위험합니다. 최소 위험도 피스 선택:")
                piece_scores = [(p, self._danger_score(self.board, self.available_pieces, p)) for p in self.available_pieces]
                piece_scores.sort(key=lambda x: x[1])
                
                for piece, score in piece_scores[:3]:  # 상위 3개만 출력
                    print(f"  {self._binary_to_mbti(piece)}: {score:.2f}")
                
                selected = piece_scores[0][0]
                print(f"\n선택된 피스: {self._binary_to_mbti(selected)} (위험도: {piece_scores[0][1]:.2f})")
                self.chosen_piece = selected  # 선택된 피스를 chosen_piece에 저장
                return selected
        
        # 정상적인 경우, place_piece에서 Minimax가 결정한 피스를 반환
        print(f"Minimax가 선택한 피스: {self._binary_to_mbti(self.chosen_piece)}")
        return self.chosen_piece

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
                score += 3
        
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
            current_player_piece: 현재 플레이어가 놓을 피스
            remaining_available_pieces: 남은 사용 가능한 피스 목록
            depth: 현재 탐색 깊이
            alpha: 알파 값 (최대화 플레이어의 최선의 값)
            beta: 베타 값 (최소화 플레이어의 최선의 값)
            is_maximizing_player: True면 P1(최대화), False면 P2(최소화)의 턴
        Returns:
            Tuple[float, Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int]]]: 
            (평가 점수, 상대에게 줄 피스, 놓을 위치)
        """
        # 게임 종료 체크
        if self._check_win_cached(board):
            return (float('-inf') if is_maximizing_player else float('inf'), 
                   remaining_available_pieces[0] if remaining_available_pieces else None,
                   self._get_empty_positions(list(map(list, board)))[0] if self._get_empty_positions(list(map(list, board))) else None)

        # 깊이 제한 또는 사용 가능한 피스가 없는 경우
        if depth == 0 or not remaining_available_pieces:
            eval_score = self._evaluate_game_state(board)
            return (eval_score,
                   remaining_available_pieces[0] if remaining_available_pieces else None,
                   self._get_empty_positions(list(map(list, board)))[0] if self._get_empty_positions(list(map(list, board))) else None)

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

            for r, c, piece_to_give in ordered_moves:
                temp_board_after_place = [list(row) for row in board]
                temp_board_after_place[r][c] = self.piece_to_index[current_player_piece]
                
                # 내가 이 수로 승리하는지 체크
                if self._check_win(temp_board_after_place):
                    return (float('inf'), piece_to_give, (r,c))

                # 상대가 이 피스로 이길 수 있는지 체크
                if self._is_immediate_win_for_opponent(temp_board_after_place, list(remaining_available_pieces), piece_to_give):
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

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_pos_to_place = (r, c)
                    best_piece_to_give = piece_to_give

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            return (max_eval, best_piece_to_give, best_pos_to_place)

        else:  # P2의 턴
            min_eval = float('inf')
            best_pos_to_place = None
            best_piece_to_give = None

            ordered_moves = self._get_ordered_moves(
                list(map(list, board)),
                list(remaining_available_pieces),
                current_player_piece
            )

            for r, c, piece_to_give in ordered_moves:
                temp_board_after_place = [list(row) for row in board]
                temp_board_after_place[r][c] = self.piece_to_index[current_player_piece]

                if self._check_win(temp_board_after_place):
                    return (float('-inf'), piece_to_give, (r,c))

                if self._is_immediate_win_for_opponent(temp_board_after_place, list(remaining_available_pieces), piece_to_give):
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
        empty_positions = self._get_empty_positions(board)
        
        # 1. 승리 가능한 수를 먼저 평가
        if piece_to_place:
            for r, c in empty_positions:
                if self._is_winning_move(board, r, c, piece_to_place):
                    return [(r, c, piece_to_place)]  # 승리 수가 있으면 즉시 반환
        
        # 2. 상대방의 즉시 승리를 막는 수를 다음으로 평가
        blocking_moves = []
        for r, c in empty_positions:
            for piece in current_available_pieces:
                temp_board = [row.copy() for row in board]
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
                if (r, c, piece) not in moves and self._has_fork_opportunity(board, current_available_pieces, piece, (r, c)):
                    fork_moves.append((r, c, piece))
        
        if fork_moves:
            moves.extend(fork_moves)
        
        # 4. 나머지 수들을 평가 점수 순으로 정렬
        remaining_moves = []
        for r, c in empty_positions:
            for piece in current_available_pieces:
                if (r, c, piece) not in moves:
                    score = self._evaluate_position(board, (r, c), piece)
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
        temp_board = [row.copy() for row in board]
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
                                    print(f"2x2블록 승리: ({r},{c})에 {self._binary_to_mbti(piece)} 놓으면 승리")
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

    @lru_cache(maxsize=16384)
    def _evaluate_game_state(self, board: Tuple[Tuple[int, ...], ...]) -> float:
        """
        주어진 보드 상태를 P2(최소화 플레이어)의 관점에서 평가합니다.
        점수가 높을수록 P2에게 유리한 상태를 나타냅니다.
        
        Args:
            board: 평가할 게임 보드 상태 (튜플 형태)
        Returns:
            float: 평가 점수
        """
        if self.debug:
            sys.stdout.write(f"  [Evaluate] 보드 상태 평가 시작...\n")
            sys.stdout.flush()

        # 1. 게임 종료 조건 확인 (가장 높은/낮은 우선순위)
        # P2의 승리 (나의 승리)
        if any(self._is_winning_move(list(map(list, board)), r, c, self.index_to_piece[board[r][c]]) 
               for r in range(4) for c in range(4) if board[r][c] != 0 and board[r][c] % 2 == 0):  # P2의 피스
            if self.debug:
                sys.stdout.write(f"  [Evaluate] P2 승리 감지! Score: {self.WIN_SCORE}\n")
                sys.stdout.flush()
            return self.WIN_SCORE
        
        # P1의 승리 (상대방의 승리)
        if any(self._is_winning_move(list(map(list, board)), r, c, self.index_to_piece[board[r][c]]) 
               for r in range(4) for c in range(4) if board[r][c] != 0 and board[r][c] % 2 != 0):  # P1의 피스
            if self.debug:
                sys.stdout.write(f"  [Evaluate] P1 승리 감지! Score: {self.LOSE_SCORE}\n")
                sys.stdout.flush()
            return self.LOSE_SCORE
        
        # 무승부 (보드가 꽉 찼고 승리 조건 없음)
        if all(cell != 0 for row in board for cell in row):
            if self.debug:
                sys.stdout.write(f"  [Evaluate] 무승부 감지! Score: {self.DRAW_SCORE}\n")
                sys.stdout.flush()
            return self.DRAW_SCORE

        score = 0.0

        # 2. 중앙 제어 및 코너 제어 (P2의 피스에 가중치)
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        corner_positions = [(0,0), (0,3), (3,0), (3,3)]

        for r, c in center_positions:
            if board[r][c] != 0:
                # P2의 피스 (짝수 인덱스)는 긍정적, P1의 피스 (홀수 인덱스)는 부정적
                score += (1 if board[r][c] % 2 == 0 else -1) * self.CENTER_BONUS
        
        for r, c in corner_positions:
            if board[r][c] != 0:
                score += (1 if board[r][c] % 2 == 0 else -1) * self.CORNER_BONUS

        # 3. 잠재적 라인 평가
        # 각 빈 칸에 대해 P2가 놓을 수 있는 피스들로 평가
        for r, c in self._get_empty_positions(list(map(list, board))):
            # P2가 놓을 수 있는 피스들로 이 위치에 놓았을 때의 잠재력
            for piece in self.available_pieces:
                if self.debug:
                    sys.stdout.write(f"  [Evaluate] 위치 ({r},{c}), 피스 {self._binary_to_mbti(piece)} (idx {self.piece_to_index[piece]}) 에 대한 잠재적 라인 계산...\n")
                    sys.stdout.flush()
                
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

        if self.debug:
            sys.stdout.write(f"  [Evaluate] 최종 평가 점수: {score}\n")
            sys.stdout.flush()
        return score

    def get_chosen_piece(self) -> Tuple[int,int,int,int]:
        """
        P2가 선택한 피스를 반환합니다.
        이 피스는 P1에게 전달되어 P1의 다음 턴에 사용됩니다.
        
        Returns:
            Tuple[int,int,int,int]: P2가 선택한 피스
        """
        if not hasattr(self, 'chosen_piece') or self.chosen_piece is None:
            print("⚠️ Warning: chosen_piece가 설정되지 않았습니다. 기본값 반환.")
            return (0,0,0,0)
        
        print(f"\n===== [P2] 선택한 피스 반환 =====")
        print(f"선택한 피스: {self._binary_to_mbti(self.chosen_piece)}")
        
        # 선택한 피스를 available_pieces에서 제거
        if self.chosen_piece in self.available_pieces:
            self.available_pieces.remove(self.chosen_piece)
            print(f"남은 피스 수: {len(self.available_pieces)}")
        else:
            print("⚠️ Warning: 선택한 피스가 available_pieces에 없습니다.")
        
        return self.chosen_piece