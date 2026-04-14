"""Pygame Reversi GUI with move list and navigation."""

# Pylint doesn't understand many dynamic pygame attributes/constants.
# pylint: disable=no-member,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from reversi_engine import (
    BLACK as ENGINE_BLACK,
    MCTS,
    PolicyValueNet,
    ReversiState,
    WHITE as ENGINE_WHITE,
    apply_action as engine_apply_action,
)


# ============================================================
# Configuration
# ============================================================
BOARD_PIXELS = 640
PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_PIXELS + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_PIXELS

BOARD_SIZE = 8
SQ_SIZE = BOARD_PIXELS // BOARD_SIZE
FPS = 60

BOARD_LIGHT = (22, 112, 54)
BOARD_DARK = (18, 92, 45)
GRID = (10, 55, 25)
HIGHLIGHT = (255, 214, 102)
BG = (26, 26, 26)

PANEL_BG = (38, 40, 44)
PANEL_BORDER = (68, 72, 78)
TEXT = (235, 235, 235)
SUBTEXT = (180, 180, 180)
ACCENT = (86, 156, 214)
ACCENT_DARK = (58, 112, 168)
BUTTON_DISABLED = (80, 80, 80)
ROW_HIGHLIGHT = (60, 92, 126)
BLACK_DISK = (30, 30, 30)
WHITE_DISK = (245, 245, 245)
SHADOW = (0, 0, 0, 55)

EMPTY = 0
BLACK = 1
WHITE = -1
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
FILES = 'abcdefgh'
WEIGHTS_FILE = Path(__file__).with_name('reversi_engine_weights.npz')


@dataclass
class GameState:
    board: List[List[int]]
    current_player: int
    move_label: Optional[str] = None


class ReversiGUI:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Interactive Reversi GUI using pygame for rendering and input."""

    def __init__(self) -> None:
        pygame.display.init()
        pygame.font.init()

        pygame.display.set_caption('2D Reversi Board')
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont('arial', 18)
        self.font_body = pygame.font.SysFont('arial', 20)
        self.font_title = pygame.font.SysFont('arial', 26, bold=True)
        self.font_status = pygame.font.SysFont('arial', 22, bold=True)
        self.font_disk = pygame.font.SysFont('arial', 20, bold=True)

        self.history: List[GameState] = [GameState(self.initial_board(), BLACK, None)]
        self.current_index = 0
        self.selected_square: Optional[Tuple[int, int]] = None
        self.engine_enabled = False
        self.engine_ready = False
        self.engine_status = 'Engine unavailable (no trained weights)'
        self.engine_cache_key: Optional[Tuple[int, int]] = None
        self.engine_lines: List[Tuple[List[str], float]] = []

        self.engine_net = PolicyValueNet(hidden_size=128, learning_rate=0.01, seed=42)
        self.engine_mcts = MCTS(self.engine_net, simulations=64)
        self._try_load_engine_weights()

        panel_x = BOARD_PIXELS
        margin = 16
        btn_w = (PANEL_WIDTH - 3 * margin) // 2
        btn_h = 42
        btn_y = WINDOW_HEIGHT - btn_h - margin

        self.back_button = pygame.Rect(panel_x + margin, btn_y, btn_w, btn_h)
        self.forward_button = pygame.Rect(panel_x + 2 * margin + btn_w, btn_y, btn_w, btn_h)
        self.engine_button = pygame.Rect(panel_x + margin, btn_y - btn_h - 10, PANEL_WIDTH - 2 * margin, btn_h)

    # --------------------------------------------------------
    # Core helpers
    # --------------------------------------------------------
    @staticmethod
    def initial_board() -> List[List[int]]:
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[3][3] = WHITE
        board[3][4] = BLACK
        board[4][3] = BLACK
        board[4][4] = WHITE
        return board

    @staticmethod
    def copy_board(board: List[List[int]]) -> List[List[int]]:
        return [row[:] for row in board]

    def current_state(self) -> GameState:
        return self.history[self.current_index]

    def live_state(self) -> GameState:
        return self.history[-1]

    def is_live_view(self) -> bool:
        return self.current_index == len(self.history) - 1

    def close(self) -> None:
        pygame.quit()

    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def square_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        return col * SQ_SIZE, row * SQ_SIZE

    def screen_to_square(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        if not (0 <= x < BOARD_PIXELS and 0 <= y < BOARD_PIXELS):
            return None
        return y // SQ_SIZE, x // SQ_SIZE

    def valid_moves(self, board: List[List[int]], player: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        moves: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != EMPTY:
                    continue

                flips: List[Tuple[int, int]] = []
                for dr, dc in DIRECTIONS:
                    r = row + dr
                    c = col + dc
                    line: List[Tuple[int, int]] = []

                    while self.in_bounds(r, c) and board[r][c] == -player:
                        line.append((r, c))
                        r += dr
                        c += dc

                    if line and self.in_bounds(r, c) and board[r][c] == player:
                        flips.extend(line)

                if flips:
                    moves[(row, col)] = flips

        return moves

    def counts(self, board: List[List[int]]) -> Tuple[int, int]:
        black = sum(cell == BLACK for row in board for cell in row)
        white = sum(cell == WHITE for row in board for cell in row)
        return black, white

    def is_game_over(self, state: Optional[GameState] = None) -> bool:
        active = state or self.current_state()
        board = active.board
        return not self.valid_moves(board, BLACK) and not self.valid_moves(board, WHITE)

    def game_over_text(self, state: Optional[GameState] = None) -> str:
        active = state or self.current_state()
        black_count, white_count = self.counts(active.board)
        if black_count > white_count:
            return f'Game over - Black wins {black_count}-{white_count}'
        if white_count > black_count:
            return f'Game over - White wins {white_count}-{black_count}'
        return f'Game over - Draw {black_count}-{white_count}'

    def move_to_label(self, row: int, col: int) -> str:
        return f'{FILES[col]}{row + 1}'

    def _try_load_engine_weights(self) -> None:
        try:
            data = np.load(WEIGHTS_FILE)
            self.engine_net.w1 = data['w1']
            self.engine_net.b1 = data['b1']
            self.engine_net.wp = data['wp']
            self.engine_net.bp = data['bp']
            self.engine_net.wv = data['wv']
            self.engine_net.bv = data['bv']
            self.engine_ready = True
            self.engine_status = 'Engine ready'
        except (OSError, KeyError, ValueError):
            self.engine_ready = False
            self.engine_enabled = False
            self.engine_status = f'Engine unavailable ({WEIGHTS_FILE.name} not found)'

    def _state_to_engine(self, state: GameState) -> ReversiState:
        board_np = np.array(state.board, dtype=np.int8)
        current_player = ENGINE_BLACK if state.current_player == BLACK else ENGINE_WHITE
        return ReversiState(board=board_np, current_player=current_player)

    def _action_to_label(self, action: int) -> str:
        if action >= BOARD_SIZE * BOARD_SIZE:
            return 'pass'
        row, col = divmod(action, BOARD_SIZE)
        return self.move_to_label(row, col)

    def _top_actions(self, state: ReversiState, limit: int) -> List[int]:
        policy = self.engine_mcts.run(state, temperature=1.0)
        actions: List[int] = []
        for action in np.argsort(policy)[::-1]:
            action_int = int(action)
            if policy[action_int] <= 0.0:
                continue
            if action_int < BOARD_SIZE * BOARD_SIZE and state.board.flat[action_int] != EMPTY:
                continue
            actions.append(action_int)
            if len(actions) >= limit:
                break
        return actions

    def _build_engine_lines(
        self, state: GameState, line_count: int = 3, depth: int = 5
    ) -> List[Tuple[List[str], float]]:
        root = self._state_to_engine(state)
        root_player = root.current_player
        lines: List[Tuple[List[str], float]] = []

        for first_action in self._top_actions(root, limit=line_count):
            line_actions: List[str] = []
            node = root
            action = first_action
            for _ in range(depth):
                line_actions.append(self._action_to_label(action))
                node = engine_apply_action(node, action)
                next_actions = self._top_actions(node, limit=1)
                if not next_actions:
                    break
                action = next_actions[0]

            _, value = self.engine_net.predict(node)
            line_eval = value if node.current_player == root_player else -value
            lines.append((line_actions, line_eval))

        return lines

    def refresh_engine_hint(self) -> None:
        state = self.current_state()
        cache_key = (self.current_index, state.current_player)
        if self.engine_cache_key == cache_key:
            return

        self.engine_cache_key = cache_key
        self.engine_lines = []

        if not self.engine_enabled or not self.engine_ready or self.is_game_over(state):
            return

        legal_moves = self.valid_moves(state.board, state.current_player)
        if not legal_moves:
            self.engine_status = 'Engine: pass'
            return

        self.engine_lines = self._build_engine_lines(state, line_count=3, depth=5)

    def push_state(self, board: List[List[int]], current_player: int, move_label: str) -> None:
        if not self.is_live_view():
            self.history = self.history[: self.current_index + 1]

        self.history.append(GameState(board, current_player, move_label))
        self.current_index = len(self.history) - 1

    def make_move(self, row: int, col: int) -> None:
        state = self.current_state()
        legal_moves = self.valid_moves(state.board, state.current_player)
        flips = legal_moves.get((row, col))
        if not flips:
            return

        new_board = self.copy_board(state.board)
        new_board[row][col] = state.current_player
        for flip_row, flip_col in flips:
            new_board[flip_row][flip_col] = state.current_player

        next_player = -state.current_player
        self.push_state(new_board, next_player, self.move_to_label(row, col))
        self.selected_square = (row, col)

        next_moves = self.valid_moves(new_board, next_player)
        current_moves = self.valid_moves(new_board, state.current_player)
        if not next_moves and current_moves:
            self.push_state(self.copy_board(new_board), state.current_player, 'pass')
        self.engine_cache_key = None

    def go_back(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.selected_square = None
            self.engine_cache_key = None

    def go_forward(self) -> None:
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.selected_square = None
            self.engine_cache_key = None

    def reset_game(self) -> None:
        self.history = [GameState(self.initial_board(), BLACK, None)]
        self.current_index = 0
        self.selected_square = None
        self.engine_cache_key = None

    # --------------------------------------------------------
    # Move list helpers
    # --------------------------------------------------------
    def get_move_rows(self) -> List[Tuple[int, str, str, int, Optional[int]]]:
        moves = [state.move_label or '' for state in self.history[1:]]
        rows = []
        i = 0
        while i < len(moves):
            move_no = i // 2 + 1
            first_move = moves[i]
            first_idx = i + 1

            if i + 1 < len(moves):
                second_move = moves[i + 1]
                second_idx = i + 2
            else:
                second_move = ''
                second_idx = None

            rows.append((move_no, first_move, second_move, first_idx, second_idx))
            i += 2

        return rows

    # --------------------------------------------------------
    # Drawing
    # --------------------------------------------------------
    def draw_board(self) -> None:
        self.screen.fill(BG)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = BOARD_LIGHT if (row + col) % 2 == 0 else BOARD_DARK
                x, y = self.square_to_screen(row, col)
                pygame.draw.rect(self.screen, color, (x, y, SQ_SIZE, SQ_SIZE))
                pygame.draw.rect(self.screen, GRID, (x, y, SQ_SIZE, SQ_SIZE), width=1)

        if self.selected_square is not None:
            x, y = self.square_to_screen(*self.selected_square)
            pygame.draw.rect(self.screen, HIGHLIGHT, (x, y, SQ_SIZE, SQ_SIZE), width=4)

        self.draw_coordinates()
        self.draw_legal_targets()

    def draw_coordinates(self) -> None:
        for col in range(BOARD_SIZE):
            label = self.font_small.render(FILES[col], True, (230, 240, 230))
            self.screen.blit(label, (col * SQ_SIZE + SQ_SIZE - 18, BOARD_PIXELS - 22))

        for row in range(BOARD_SIZE):
            label = self.font_small.render(str(row + 1), True, (230, 240, 230))
            self.screen.blit(label, (6, row * SQ_SIZE + 6))

    def draw_legal_targets(self) -> None:
        state = self.current_state()
        if self.is_game_over(state):
            return

        dot_color = BLACK_DISK if state.current_player == BLACK else WHITE_DISK
        dot_border = (220, 220, 220) if state.current_player == BLACK else (60, 60, 60)

        for row, col in self.valid_moves(state.board, state.current_player):
            x, y = self.square_to_screen(row, col)
            cx = x + SQ_SIZE // 2
            cy = y + SQ_SIZE // 2
            pygame.draw.circle(self.screen, dot_color, (cx, cy), 8)
            pygame.draw.circle(self.screen, dot_border, (cx, cy), 8, width=1)

    def draw_disk(self, row: int, col: int, color: int) -> None:
        x, y = self.square_to_screen(row, col)
        center = (x + SQ_SIZE // 2, y + SQ_SIZE // 2)
        radius = SQ_SIZE // 2 - 8
        disk_color = BLACK_DISK if color == BLACK else WHITE_DISK
        rim_color = (220, 220, 220) if color == BLACK else (60, 60, 60)

        shadow = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(shadow, SHADOW, (SQ_SIZE // 2 + 2, SQ_SIZE // 2 + 4), radius)
        self.screen.blit(shadow, (x, y))

        pygame.draw.circle(self.screen, disk_color, center, radius)
        pygame.draw.circle(self.screen, rim_color, center, radius, width=2)

    def draw_pieces(self) -> None:
        board = self.current_state().board
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != EMPTY:
                    self.draw_disk(row, col, board[row][col])

    def draw_game_over(self) -> None:
        state = self.current_state()
        if not self.is_game_over(state):
            return

        overlay = pygame.Surface((BOARD_PIXELS, BOARD_PIXELS), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        self.screen.blit(overlay, (0, 0))

        text = self.font_status.render(self.game_over_text(state), True, (255, 255, 255))
        rect = text.get_rect(center=(BOARD_PIXELS // 2, BOARD_PIXELS // 2))
        self.screen.blit(text, rect)

    # --------------------------------------------------------
    # Side panel drawing
    # --------------------------------------------------------
    def draw_button(self, rect: pygame.Rect, label: str, enabled: bool) -> None:
        color = ACCENT if enabled else BUTTON_DISABLED
        border = ACCENT_DARK if enabled else (60, 60, 60)

        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=8)

        txt = self.font_body.render(label, True, (255, 255, 255))
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    def draw_side_panel(self) -> None:  # pylint: disable=too-many-locals,too-many-statements
        state = self.current_state()
        self.refresh_engine_hint()
        panel_rect = pygame.Rect(BOARD_PIXELS, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(
            self.screen, PANEL_BORDER, (BOARD_PIXELS, 0), (BOARD_PIXELS, WINDOW_HEIGHT), 2
        )

        panel_x = BOARD_PIXELS + 16
        y = 16

        title = self.font_title.render('Moves', True, TEXT)
        self.screen.blit(title, (panel_x, y))
        y += 40

        if self.current_index == 0:
            status_text = 'Start position'
        elif self.is_live_view():
            status_text = f'Live position ({self.current_index} turns)'
        else:
            status_text = f'History view ({self.current_index}/{len(self.history) - 1} turns)'

        status = self.font_body.render(status_text, True, SUBTEXT)
        self.screen.blit(status, (panel_x, y))
        y += 34

        black_count, white_count = self.counts(state.board)
        score_text = f'Black {black_count}  White {white_count}'
        score_surface = self.font_body.render(score_text, True, TEXT)
        self.screen.blit(score_surface, (panel_x, y))
        y += 30

        if self.is_game_over(state):
            turn_text = self.game_over_text(state)
        else:
            turn_name = 'Black' if state.current_player == BLACK else 'White'
            legal_count = len(self.valid_moves(state.board, state.current_player))
            turn_text = f'{turn_name} to move ({legal_count} legal)'

        turn_surface = self.font_body.render(turn_text, True, TEXT)
        self.screen.blit(turn_surface, (panel_x, y))
        y += 26

        if self.engine_enabled and self.engine_ready and not self.is_game_over(state):
            if not self.engine_lines:
                self.screen.blit(self.font_small.render('Lines: pass', True, TEXT), (panel_x, y))
                y += 20
            else:
                for idx, (line_moves, line_eval) in enumerate(self.engine_lines, start=1):
                    line_text = ' '.join(line_moves)
                    render_text = f'{idx}) {line_text}  [{line_eval:+.3f}]'
                    self.screen.blit(self.font_small.render(render_text, True, TEXT), (panel_x, y))
                    y += 18
            y += 8
        else:
            y += 8

        pygame.draw.line(
            self.screen,
            PANEL_BORDER,
            (BOARD_PIXELS + 12, y + 10),
            (WINDOW_WIDTH - 12, y + 10),
            1,
        )
        y += 24

        rows = self.get_move_rows()
        row_height = 28
        available_height = self.back_button.top - y - 10
        max_rows = max(1, available_height // row_height)

        current_row = 0 if self.current_index == 0 else (self.current_index - 1) // 2
        start_row = max(0, current_row - max_rows + 3)
        end_row = min(len(rows), start_row + max_rows)

        col1 = panel_x
        col2 = panel_x + 40
        col3 = panel_x + 130

        for row_i in range(start_row, end_row):
            move_no, black_move, white_move, black_idx, white_idx = rows[row_i]
            row_y = y + (row_i - start_row) * row_height

            if self.current_index in (black_idx, white_idx):
                highlight_rect = pygame.Rect(
                    BOARD_PIXELS + 10, row_y - 2, PANEL_WIDTH - 20, row_height - 2
                )
                pygame.draw.rect(self.screen, ROW_HIGHLIGHT, highlight_rect, border_radius=6)

            num_txt = self.font_small.render(f'{move_no}.', True, SUBTEXT)
            self.screen.blit(num_txt, (col1, row_y + 4))

            black_color = TEXT if black_idx == self.current_index else (220, 220, 220)
            white_color = TEXT if white_idx == self.current_index else (220, 220, 220)

            black_txt = self.font_body.render(black_move, True, black_color)
            self.screen.blit(black_txt, (col2, row_y + 2))

            if white_move:
                white_txt = self.font_body.render(white_move, True, white_color)
                self.screen.blit(white_txt, (col3, row_y + 2))

        self.draw_button(self.back_button, 'Back', self.current_index > 0)
        self.draw_button(
            self.forward_button, 'Forward', self.current_index < len(self.history) - 1
        )
        if self.engine_ready:
            engine_label = 'Engine: ON' if self.engine_enabled else 'Engine: OFF'
            self.draw_button(self.engine_button, engine_label, True)
        else:
            self.draw_button(self.engine_button, 'Engine unavailable', False)

        help_text = self.font_small.render('Click a dot to place, R to reset', True, SUBTEXT)
        help_rect = help_text.get_rect(
            midbottom=(BOARD_PIXELS + PANEL_WIDTH // 2, self.engine_button.top - 10)
        )
        self.screen.blit(help_text, help_rect)

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------
    def try_place_disk(self, pos: Tuple[int, int]) -> None:
        if self.is_game_over():
            return

        square = self.screen_to_square(pos)
        if square is None:
            return

        row, col = square
        self.make_move(row, col)

    def handle_mouse_down(self, pos: Tuple[int, int]) -> None:
        if self.engine_button.collidepoint(pos) and self.engine_ready:
            self.engine_enabled = not self.engine_enabled
            self.engine_cache_key = None
            return

        if self.back_button.collidepoint(pos) and self.current_index > 0:
            self.go_back()
            return

        if self.forward_button.collidepoint(pos) and self.current_index < len(self.history) - 1:
            self.go_forward()
            return

        if pos[0] < BOARD_PIXELS:
            self.try_place_disk(pos)

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    def run(self) -> None:  # pylint: disable=too-many-branches
        running = True
    
        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_LEFT:
                        self.go_back()
                    elif event.key == pygame.K_RIGHT:
                        self.go_forward()
                    elif event.key == pygame.K_r:
                        self.reset_game()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse_down(event.pos)

            self.draw_board()
            self.draw_pieces()
            self.draw_game_over()
            self.draw_side_panel()
            pygame.display.flip()

        self.close()


def main() -> None:
    """Run the Reversi GUI application."""
    ReversiGUI().run()


if __name__ == '__main__':
    main()
