import random
import copy
from typing import List, Tuple

class P2:
    def __init__(self, board: List[List[int]], available_pieces: List[Tuple[int,int,int,int]]):
        # 깊은 복사로 안전하게 초기 상태 저장
        self.board = [row.copy() for row in board]
        self.available_pieces = available_pieces.copy()
        # 모든 16개 피스 조합 생성 및 인덱스 매핑
        self.pieces = [(i,j,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.piece_to_index = {p: idx+1 for idx,p in enumerate(self.pieces)}

    def select_piece(self, simulations: int = 500) -> Tuple[int,int,int,int]:
        """
        상대에게 넘길 피스를 몬테카를로 시뮬레이션으로 평가
        simulations: 각 피스별 시뮬레이션 횟수
        """
        scores = {}
        for piece in self.available_pieces:
            lose_count = 0
            for _ in range(simulations):
                if self._simulate_after_give(piece):
                    lose_count += 1
            scores[piece] = lose_count
        # 가장 낮은 패배(상대 승리) 횟수 피스 선택
        min_score = min(scores.values())
        best = [p for p,s in scores.items() if s == min_score]
        return random.choice(best)

    def place_piece(self, selected_piece: Tuple[int,int,int,int], simulations: int = 100) -> Tuple[int,int]:
        """
        받은 피스를 놓을 위치를 몬테카를로 시뮬레이션으로 평가
        """
        idx = self.piece_to_index[selected_piece]
        empty = self._get_empty_positions(self.board)
        # 즉시 승리 위치 우선
        for r,c in empty:
            temp = [row.copy() for row in self.board]
            temp[r][c] = idx
            if self._check_win(temp):
                return (r,c)
        # 시뮬레이션으로 최적 위치 선택
        win_counts = {pos:0 for pos in empty}
        for pos in empty:
            for _ in range(simulations):
                if self._simulate_after_place(selected_piece, pos):
                    win_counts[pos] += 1
        max_win = max(win_counts.values())
        best = [pos for pos,s in win_counts.items() if s == max_win]
        return random.choice(best)

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
        # 받은 피스를 pos에 배치 후 우리(현재 플레이어)가 승리하는지 시뮬레이션
        board = [row.copy() for row in self.board]
        avail = self.available_pieces.copy()
        avail.remove(piece)
        board[pos[0]][pos[1]] = self.piece_to_index[piece]
        if self._check_win(board):
            return True
        turn = 1  # 다음 시뮬레이션부터 상대 배치 순서
        # 무작위 시뮬레이션
        while avail:
            nxt = random.choice(avail)
            avail.remove(nxt)
            r,c = random.choice(self._get_empty_positions(board))
            board[r][c] = self.piece_to_index[nxt]
            if self._check_win(board):
                return turn == 0  # 우리가 배치할 때 승리해야 True
            turn ^= 1
        return False

    def _get_empty_positions(self, board: List[List[int]]) -> List[Tuple[int,int]]:
        return [(r,c) for r in range(4) for c in range(4) if board[r][c] == 0]

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
        if 0 in line: return False
        attrs = [self.pieces[idx-1] for idx in line]
        for i in range(4):
            if len({a[i] for a in attrs}) == 1:
                return True
        return False

    def _check_2x2(self, block: List[int]) -> bool:
        attrs = [self.pieces[idx-1] for idx in block]
        for i in range(4):
            if len({a[i] for a in attrs}) == 1:
                return True
        return False