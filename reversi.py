"""Pygame Reversi GUI with move list and navigation.

Run: python reversi.py · Train / save weights: python reversi_engine.py
(weights file: reversi_engine_weights.npz next to these modules.)
"""

# Pylint doesn't understand many dynamic pygame attributes/constants.
# pylint: disable=no-member,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pygame

from reversi_engine import (
    BLACK as ENGINE_BLACK,
    EMPTY as ENGINE_EMPTY,
    MCTS,
    PolicyValueNet,
    ReversiState,
    WHITE as ENGINE_WHITE,
    apply_action as engine_apply_action,
    is_terminal as engine_is_terminal,
    PASS_MOVE as ENGINE_PASS_MOVE,
    winner as engine_winner,
)


# ============================================================
# Configuration
# ============================================================
BOARD_PIXELS = 640
PANEL_WIDTH = 400
WINDOW_WIDTH = BOARD_PIXELS + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_PIXELS
BOARD_SIZE = 8
SQ_SIZE = BOARD_PIXELS // BOARD_SIZE
FPS = 60
FLIP_ANIM_MS = 320

BOARD_LIGHT = (22, 112, 54)
BOARD_DARK = (18, 92, 45)
GRID = (10, 55, 25)
HIGHLIGHT = (255, 214, 102)
BG = (26, 26, 26)

PANEL_BG = (38, 40, 44)
PANEL_BORDER = (68, 72, 78)
SCROLLBAR_WIDTH = 10
SCROLLBAR_PAD = 4
SCROLLBAR_TRACK = (52, 54, 58)
SCROLLBAR_THUMB = (120, 125, 135)
SCROLLBAR_THUMB_HOVER = (150, 155, 165)
# Cap move list height so it does not crowd buttons below.
MOVES_LIST_MAX_HEIGHT = 168
MOVES_LIST_BOTTOM_GAP = 32
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
# MCTS budgets (tuned for “deeper but runnable” hints on a laptop CPU):
# - One strong eval at the root; lighter per-step policy along lines; modest line-end eval.
# - Line ply depth grows in midgame/endgame (fewer legal moves → cheaper per ply).
ENGINE_MCTS_SIMULATIONS_PLAY = 96
ENGINE_MCTS_SIMULATIONS_HINTS = 22
ENGINE_MCTS_SIMULATIONS_EVAL_POSITION = 120
ENGINE_MCTS_SIMULATIONS_EVAL_LINE = 56
ENGINE_HINT_LINE_COUNT = 3
ENGINE_HINT_LINE_DEPTH = 8
ENGINE_HINT_LINE_DEPTH_MID = 10
ENGINE_HINT_LINE_DEPTH_ENDGAME = 12
ENGINE_EMPTY_MID = 38
ENGINE_EMPTY_END = 22

EVAL_BAR_HEIGHT = 22
EVAL_BAR_BORDER = (88, 92, 98)
EVAL_BAR_MID_TICK = (120, 125, 130)
# Exponential smoothing toward new MCTS values (higher = snappier, frame-rate independent).
EVAL_BAR_SMOOTH_SPEED = 7.5


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
        self.engine_status = ''
        self.engine_cache_key: Optional[Tuple[int, int]] = None
        self.engine_lines: List[Tuple[List[str], float]] = []
        self.engine_position_white_eval: Optional[float] = None
        self.engine_bar_white_eval: Optional[float] = None
        self._engine_bar_display_eval: Optional[float] = None
        self.play_vs_engine = False
        self._vs_engine_delay_until: Optional[int] = None

        self.flip_anim_for_index: Optional[int] = None
        self.flip_anim_cells: List[Tuple[int, int, int, int]] = []
        self.flip_anim_accum_ms: float = 0.0
        self.flip_anim_last_tick: int = 0

        self.engine_net: PolicyValueNet
        self.engine_mcts: MCTS
        self.engine_mcts_hints: MCTS
        self._init_engine_network()

        panel_x = BOARD_PIXELS
        margin = 16
        btn_w = (PANEL_WIDTH - 3 * margin) // 2
        btn_h = 42
        btn_y = WINDOW_HEIGHT - btn_h - margin

        self.back_button = pygame.Rect(panel_x + margin, btn_y, btn_w, btn_h)
        self.forward_button = pygame.Rect(panel_x + 2 * margin + btn_w, btn_y, btn_w, btn_h)
        btn_gap = 8
        self.engine_button = pygame.Rect(
            panel_x + margin, btn_y - btn_h - btn_gap, PANEL_WIDTH - 2 * margin, btn_h
        )
        self.play_vs_engine_button = pygame.Rect(
            panel_x + margin,
            self.engine_button.top - btn_h - btn_gap,
            PANEL_WIDTH - 2 * margin,
            btn_h,
        )

        self.moves_scroll = 0
        self.moves_scroll_dragging = False
        self._scroll_drag_s0 = 0
        self._scroll_drag_my0 = 0

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

    def _init_engine_network(self) -> None:
        """Load PolicyValueNet (CNN AlphaZero-style or legacy MLP) from checkpoint."""
        try:
            with np.load(WEIGHTS_FILE) as data:
                self.engine_net = PolicyValueNet.from_npz(data)
            self.engine_mcts = MCTS(self.engine_net, simulations=ENGINE_MCTS_SIMULATIONS_PLAY)
            self.engine_mcts_hints = MCTS(self.engine_net, simulations=ENGINE_MCTS_SIMULATIONS_HINTS)
            self.engine_ready = True
            self.engine_status = ''
        except OSError as exc:
            self.engine_net = PolicyValueNet(arch='cnn', conv_channels=64, learning_rate=0.01, seed=42)
            self.engine_mcts = MCTS(self.engine_net, simulations=64)
            self.engine_mcts_hints = MCTS(self.engine_net, simulations=ENGINE_MCTS_SIMULATIONS_HINTS)
            self.engine_ready = False
            self.engine_enabled = False
            if getattr(exc, 'errno', None) == 2 or isinstance(exc, FileNotFoundError):
                self.engine_status = f'Engine unavailable (missing {WEIGHTS_FILE.name})'
            else:
                self.engine_status = f'Engine unavailable (could not read weights: {exc})'
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.engine_net = PolicyValueNet(arch='cnn', conv_channels=64, learning_rate=0.01, seed=42)
            self.engine_mcts = MCTS(self.engine_net, simulations=64)
            self.engine_mcts_hints = MCTS(self.engine_net, simulations=ENGINE_MCTS_SIMULATIONS_HINTS)
            self.engine_ready = False
            self.engine_enabled = False
            msg = str(exc).replace('\n', ' ')
            if len(msg) > 72:
                msg = msg[:69] + '...'
            self.engine_status = f'Engine unavailable (bad weights: {msg})'

    def reload_engine_weights(self) -> None:
        """Reload PolicyValueNet and MCTS from WEIGHTS_FILE (same file training saves)."""
        self._init_engine_network()
        self.engine_cache_key = None
        self.engine_lines = []
        self.engine_bar_white_eval = None
        self._engine_bar_display_eval = None
        if self.engine_ready:
            self.engine_status = ''

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
        policy = self.engine_mcts_hints.run(state, temperature=1.0)
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

    def _mcts_value_for_white(
        self, rstate: ReversiState, simulations: int
    ) -> float:
        """MCTS estimate in [-1, 1], fixed White-positive (+ favors White), any side to move."""
        v = self.engine_mcts.evaluate(rstate, simulations=simulations)
        return v if rstate.current_player == ENGINE_WHITE else -v

    def _leaf_white_eval(self, node: ReversiState) -> float:
        """Line-end score in White-positive frame; terminals use disk result, else MCTS."""
        if engine_is_terminal(node):
            win = engine_winner(node)
            if win == ENGINE_EMPTY:
                return 0.0
            return 1.0 if win == ENGINE_WHITE else -1.0
        return self._mcts_value_for_white(node, simulations=ENGINE_MCTS_SIMULATIONS_EVAL_LINE)

    @staticmethod
    def _empty_squares(state: GameState) -> int:
        return sum(cell == EMPTY for row in state.board for cell in row)

    def _hint_line_depth(self, state: GameState) -> int:
        """More plies late game (branching shrinks); cap cost early."""
        e = self._empty_squares(state)
        if e <= ENGINE_EMPTY_END:
            return ENGINE_HINT_LINE_DEPTH_ENDGAME
        if e <= ENGINE_EMPTY_MID:
            return ENGINE_HINT_LINE_DEPTH_MID
        return ENGINE_HINT_LINE_DEPTH

    def _build_engine_lines(
        self,
        state: GameState,
        line_count: int = ENGINE_HINT_LINE_COUNT,
        position_white: Optional[float] = None,
    ) -> Tuple[List[Tuple[List[str], float]], float]:
        root = self._state_to_engine(state)
        root_player = root.current_player
        depth = self._hint_line_depth(state)
        if position_white is None:
            position_white = self._mcts_value_for_white(
                root, simulations=ENGINE_MCTS_SIMULATIONS_EVAL_POSITION
            )
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

            lines.append((line_actions, self._leaf_white_eval(node)))

        # Best line for side to move: maximize White score if White moves, minimize if Black moves.
        lines.sort(key=lambda item: item[1], reverse=(root_player == ENGINE_WHITE))
        return lines, position_white

    def refresh_engine_hint(self) -> None:
        state = self.current_state()
        cache_key = (self.current_index, state.current_player)
        if self.engine_cache_key == cache_key:
            return

        self.engine_cache_key = cache_key
        self.engine_lines = []
        self.engine_position_white_eval = None
        self.engine_bar_white_eval = None

        if not self.engine_ready:
            return

        if self.is_game_over(state):
            black_count, white_count = self.counts(state.board)
            if white_count > black_count:
                end_ev = 1.0
            elif black_count > white_count:
                end_ev = -1.0
            else:
                end_ev = 0.0
            self.engine_position_white_eval = end_ev
            self.engine_bar_white_eval = end_ev
            return

        root = self._state_to_engine(state)
        position_white = self._mcts_value_for_white(
            root, simulations=ENGINE_MCTS_SIMULATIONS_EVAL_POSITION
        )
        self.engine_position_white_eval = position_white
        self.engine_bar_white_eval = position_white

        if not self.engine_enabled or self.play_vs_engine:
            return

        legal_moves = self.valid_moves(state.board, state.current_player)
        if not legal_moves:
            self.engine_status = 'Pass'
            return

        self.engine_lines, _ = self._build_engine_lines(state, position_white=position_white)
        if self.engine_lines:
            self.engine_bar_white_eval = self.engine_lines[0][1]

    def push_state(self, board: List[List[int]], current_player: int, move_label: str) -> None:
        if not self.is_live_view():
            self.history = self.history[: self.current_index + 1]

        self.history.append(GameState(board, current_player, move_label))
        self.current_index = len(self.history) - 1

    def make_move(self, row: int, col: int) -> None:
        state = self.current_state()
        human_just_played_black = self.play_vs_engine and state.current_player == BLACK
        legal_moves = self.valid_moves(state.board, state.current_player)
        flips = legal_moves.get((row, col))
        if not flips:
            return

        self._clear_flip_animation()
        opponent = -state.current_player
        anim_cells: List[Tuple[int, int, int, int]] = [
            (fr, fc, opponent, state.current_player) for fr, fc in flips
        ]
        anim_cells.append((row, col, EMPTY, state.current_player))

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

        if human_just_played_black and self.play_vs_engine and self.is_live_view():
            if self.live_state().current_player == WHITE:
                self._vs_engine_delay_until = pygame.time.get_ticks() + 500

        self.flip_anim_cells = anim_cells
        self.flip_anim_accum_ms = 0.0
        self.flip_anim_last_tick = pygame.time.get_ticks()
        self.flip_anim_for_index = self.current_index

    def go_back(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.selected_square = None
            self.engine_cache_key = None
            self._clear_flip_animation()

    def go_forward(self) -> None:
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.selected_square = None
            self.engine_cache_key = None
            self._clear_flip_animation()

    def reset_game(self) -> None:
        self.history = [GameState(self.initial_board(), BLACK, None)]
        self.current_index = 0
        self.selected_square = None
        self.engine_cache_key = None
        self._engine_bar_display_eval = None
        self.play_vs_engine = False
        self.moves_scroll = 0
        self.moves_scroll_dragging = False
        self._vs_engine_delay_until = None
        self._clear_flip_animation()

    def _clear_flip_animation(self) -> None:
        self.flip_anim_for_index = None
        self.flip_anim_cells = []
        self.flip_anim_accum_ms = 0.0
        self.flip_anim_last_tick = 0

    def _best_engine_action(self, state: GameState) -> int:
        engine_state = self._state_to_engine(state)
        policy = self.engine_mcts.run(engine_state, temperature=0.0)
        return int(np.argmax(policy))

    def play_engine_turn(self) -> None:
        """Play one move for White when in vs-engine mode (human is Black)."""
        if not self.play_vs_engine or not self.engine_ready:
            return
        state = self.live_state()
        if state.current_player != WHITE:
            return

        legal = self.valid_moves(state.board, WHITE)
        if not legal:
            black_legal = self.valid_moves(state.board, BLACK)
            if not black_legal:
                return
            self.push_state(self.copy_board(state.board), BLACK, 'pass')
            self.engine_cache_key = None
            return

        action = self._best_engine_action(state)
        if action == ENGINE_PASS_MOVE or action >= BOARD_SIZE * BOARD_SIZE:
            row, col = next(iter(legal))
        else:
            row, col = divmod(action, BOARD_SIZE)
            if (row, col) not in legal:
                row, col = next(iter(legal))

        self.make_move(row, col)

    def maybe_play_engine_move(self) -> None:
        if not self.play_vs_engine or not self.engine_ready:
            self._vs_engine_delay_until = None
            return
        if not self.is_live_view() or self.is_game_over():
            self._vs_engine_delay_until = None
            return
        if self.live_state().current_player != WHITE:
            self._vs_engine_delay_until = None
            return
        if self._vs_engine_delay_until is not None:
            if pygame.time.get_ticks() < self._vs_engine_delay_until:
                return
            self._vs_engine_delay_until = None
        self.play_engine_turn()

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
        if self.play_vs_engine and self.is_live_view() and state.current_player == WHITE:
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

    def _disk_colors(self, color: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        disk_color = BLACK_DISK if color == BLACK else WHITE_DISK
        rim_color = (220, 220, 220) if color == BLACK else (60, 60, 60)
        return disk_color, rim_color

    def draw_disk_elliptical(self, row: int, col: int, color: int, x_scale: float) -> None:
        """Draw a disk squashed horizontally (0–1) for a flip illusion."""
        x_scale = max(0.06, min(1.0, x_scale))
        x, y = self.square_to_screen(row, col)
        cx = x + SQ_SIZE // 2
        cy = y + SQ_SIZE // 2
        radius = SQ_SIZE // 2 - 8
        rx = max(2, int(radius * x_scale))
        ry = radius
        disk_color, rim_color = self._disk_colors(color)
        rect = pygame.Rect(cx - rx, cy - ry, 2 * rx, 2 * ry)

        shadow = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        shrx = max(2, int((SQ_SIZE // 2 - 2) * x_scale))
        pygame.draw.ellipse(
            shadow, SHADOW, (SQ_SIZE // 2 - shrx + 1, SQ_SIZE // 2 - ry + 3, 2 * shrx, 2 * ry)
        )
        self.screen.blit(shadow, (x, y))

        pygame.draw.ellipse(self.screen, disk_color, rect)
        pygame.draw.ellipse(self.screen, rim_color, rect, width=2)

    def draw_disk_pop_in(self, row: int, col: int, color: int, scale: float) -> None:
        """New stone scales up from the center."""
        scale = max(0.0, min(1.0, scale))
        if scale <= 0.02:
            return
        x, y = self.square_to_screen(row, col)
        cx = x + SQ_SIZE // 2
        cy = y + SQ_SIZE // 2
        radius = int((SQ_SIZE // 2 - 8) * scale)
        if radius < 2:
            return
        disk_color, rim_color = self._disk_colors(color)

        shadow = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(shadow, SHADOW, (SQ_SIZE // 2 + 2, SQ_SIZE // 2 + 4), radius)
        self.screen.blit(shadow, (x, y))

        pygame.draw.circle(self.screen, disk_color, (cx, cy), radius)
        pygame.draw.circle(self.screen, rim_color, (cx, cy), radius, width=2)

    def _flip_animation_progress(self) -> Optional[float]:
        if self.flip_anim_for_index is None or not self.flip_anim_cells:
            return None
        if self.current_index != self.flip_anim_for_index:
            return None
        now = pygame.time.get_ticks()
        dt_raw = now - self.flip_anim_last_tick
        self.flip_anim_last_tick = now
        # Hint mode runs heavy MCTS in the same frame after draw_pieces; cap dt so one
        # long stall does not jump the flip animation to the end in a single step.
        dt = float(min(max(dt_raw, 0), 40))
        if dt == 0.0:
            dt = 17.0
        self.flip_anim_accum_ms += dt
        if self.flip_anim_accum_ms >= FLIP_ANIM_MS:
            self._clear_flip_animation()
            return None
        return self.flip_anim_accum_ms / FLIP_ANIM_MS

    def draw_piece_animated(self, row: int, col: int, from_c: int, to_c: int, t: float) -> None:
        if from_c == EMPTY:
            ease = 1.0 - (1.0 - t) ** 3
            self.draw_disk_pop_in(row, col, to_c, ease)
        else:
            squash = abs(math.cos(math.pi * t))
            color = from_c if t < 0.5 else to_c
            self.draw_disk_elliptical(row, col, color, squash)

    def draw_pieces(self) -> None:
        board = self.current_state().board
        t_anim = self._flip_animation_progress()
        anim_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        if t_anim is not None:
            anim_map = {(r, c): (fc, tc) for r, c, fc, tc in self.flip_anim_cells}

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                key = (row, col)
                if key in anim_map and t_anim is not None:
                    fc, tc = anim_map[key]
                    self.draw_piece_animated(row, col, fc, tc, t_anim)
                elif piece != EMPTY:
                    self.draw_disk(row, col, piece)

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

    def _update_engine_bar_smooth(self, dt_ms: int) -> None:
        """Ease displayed bar value toward engine_bar_white_eval (frame-delta based)."""
        target = self.engine_bar_white_eval
        if target is None:
            self._engine_bar_display_eval = None
            return
        t = float(target)
        if self._engine_bar_display_eval is None:
            self._engine_bar_display_eval = t
            return
        d = t - self._engine_bar_display_eval
        if abs(d) < 1.5e-4:
            self._engine_bar_display_eval = t
            return
        dt = max(0.0, min(float(dt_ms) / 1000.0, 0.25))
        k = 1.0 - math.exp(-EVAL_BAR_SMOOTH_SPEED * dt)
        self._engine_bar_display_eval += d * k

    def _draw_evaluation_bar(self, panel_x: int, y: int) -> int:
        """Black left, white right; 0.0 → half/half; + favors White (more white). Returns next y."""
        if self._engine_bar_display_eval is None:
            return y
        bar_w = PANEL_WIDTH - 32
        ev = float(self._engine_bar_display_eval)
        ev_c = max(-1.0, min(1.0, ev))
        white_frac = (ev_c + 1.0) * 0.5
        black_frac = 1.0 - white_frac
        split_x = panel_x + int(round(bar_w * black_frac))
        split_x = min(panel_x + bar_w, max(panel_x, split_x))
        bar_h = EVAL_BAR_HEIGHT

        if split_x > panel_x:
            pygame.draw.rect(self.screen, BLACK_DISK, (panel_x, y, split_x - panel_x, bar_h))
        if split_x < panel_x + bar_w:
            pygame.draw.rect(
                self.screen, WHITE_DISK, (split_x, y, panel_x + bar_w - split_x, bar_h)
            )
        pygame.draw.rect(
            self.screen, EVAL_BAR_BORDER, (panel_x, y, bar_w, bar_h), width=1, border_radius=4
        )
        mid_x = panel_x + bar_w // 2
        pygame.draw.line(self.screen, EVAL_BAR_MID_TICK, (mid_x, y), (mid_x, y + bar_h), 1)

        y += bar_h + 4
        self.screen.blit(self.font_small.render(f'{ev:+.3f}', True, TEXT), (panel_x, y))
        return y + 20

    def draw_side_panel(self) -> None:  # pylint: disable=too-many-locals,too-many-statements
        state = self.current_state()
        self.refresh_engine_hint()
        self._update_engine_bar_smooth(self.clock.get_time())
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

        eng_msg = self.engine_status.strip()
        if eng_msg:
            eng_color = SUBTEXT if self.engine_ready else (220, 140, 140)
            if len(eng_msg) > 70:
                eng_msg = eng_msg[:67] + '...'
            self.screen.blit(self.font_small.render(eng_msg, True, eng_color), (panel_x, y))
            y += 22

        y = self._draw_evaluation_bar(panel_x, y)

        if (
            self.engine_enabled
            and not self.play_vs_engine
            and self.engine_ready
            and not self.is_game_over(state)
        ):
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
        moves_list_top = y
        safe_available = max(0, self.back_button.top - moves_list_top - MOVES_LIST_BOTTOM_GAP)
        available_height = min(MOVES_LIST_MAX_HEIGHT, safe_available)
        if available_height < row_height:
            available_height = safe_available
        max_visible = max(1, available_height // row_height)
        total_rows = len(rows)
        max_scroll = max(0, total_rows - max_visible)
        self.moves_scroll = min(max(self.moves_scroll, 0), max_scroll)
        start_row = self.moves_scroll
        end_row = min(total_rows, start_row + max_visible)

        list_clip_w = PANEL_WIDTH - 24 - SCROLLBAR_WIDTH - SCROLLBAR_PAD
        clip_rect = pygame.Rect(BOARD_PIXELS + 12, moves_list_top, list_clip_w, available_height)
        track_x = clip_rect.right + SCROLLBAR_PAD
        track_rect = pygame.Rect(track_x, moves_list_top, SCROLLBAR_WIDTH, available_height)

        self._moves_list_clip_rect = clip_rect
        self._scrollbar_track_rect = track_rect
        self._scrollbar_max_scroll = max_scroll
        self._max_visible_moves = max_visible

        thumb_h = available_height
        if max_scroll > 0 and total_rows > 0:
            thumb_h = max(24, int(available_height * max_visible / total_rows))
        thumb_h = min(thumb_h, available_height)
        span = max(1, available_height - thumb_h)
        thumb_y = track_rect.y
        if max_scroll > 0:
            thumb_y = track_rect.y + int((self.moves_scroll / max_scroll) * span)
        thumb_rect = pygame.Rect(track_rect.x, thumb_y, SCROLLBAR_WIDTH, thumb_h)
        self._scrollbar_thumb_rect = thumb_rect
        self._scrollbar_thumb_h = thumb_h
        self._scrollbar_span = span

        pygame.draw.rect(self.screen, SCROLLBAR_TRACK, track_rect, border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, track_rect, width=1, border_radius=4)
        pygame.draw.rect(self.screen, SCROLLBAR_THUMB, thumb_rect, border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, thumb_rect, width=1, border_radius=4)

        col1 = panel_x
        col2 = panel_x + 40
        col3 = panel_x + 130

        old_clip = self.screen.get_clip()
        self.screen.set_clip(clip_rect)
        for row_i in range(start_row, end_row):
            move_no, black_move, white_move, black_idx, white_idx = rows[row_i]
            row_y = moves_list_top + (row_i - start_row) * row_height

            if self.current_index in (black_idx, white_idx):
                highlight_rect = pygame.Rect(
                    clip_rect.x, row_y - 2, clip_rect.width, row_height - 2
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

        self.screen.set_clip(old_clip)

        self.draw_button(self.back_button, 'Back', self.current_index > 0)
        self.draw_button(
            self.forward_button, 'Forward', self.current_index < len(self.history) - 1
        )
        if self.engine_ready:
            play_label = 'Vs engine: ON' if self.play_vs_engine else 'Vs engine: OFF'
            self.draw_button(self.play_vs_engine_button, play_label, True)
            if self.play_vs_engine:
                self.draw_button(self.engine_button, 'Hints: off (vs mode)', False)
            else:
                engine_label = 'Hints: ON' if self.engine_enabled else 'Hints: OFF'
                self.draw_button(self.engine_button, engine_label, True)
        else:
            self.draw_button(self.play_vs_engine_button, 'Play vs engine', False)
            self.draw_button(self.engine_button, 'Engine unavailable', False)

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------
    def handle_moves_scrollbar_mouse_down(self, pos: Tuple[int, int]) -> bool:
        if not hasattr(self, '_scrollbar_track_rect') or self._scrollbar_track_rect.width <= 0:
            return False
        if not self._scrollbar_track_rect.collidepoint(pos):
            return False
        if self._scrollbar_max_scroll <= 0:
            return True

        if self._scrollbar_thumb_rect.collidepoint(pos):
            self.moves_scroll_dragging = True
            self._scroll_drag_s0 = self.moves_scroll
            self._scroll_drag_my0 = pos[1]
        else:
            page = max(1, self._max_visible_moves)
            if pos[1] < self._scrollbar_thumb_rect.centery:
                self.moves_scroll = max(0, self.moves_scroll - page)
            else:
                self.moves_scroll = min(self._scrollbar_max_scroll, self.moves_scroll + page)
        return True

    def update_moves_scroll_drag(self) -> None:
        if not self.moves_scroll_dragging:
            return
        if not pygame.mouse.get_pressed()[0]:
            self.moves_scroll_dragging = False
            return
        max_scroll = self._scrollbar_max_scroll
        if max_scroll <= 0:
            self.moves_scroll_dragging = False
            return
        span = self._scrollbar_span
        dy = pygame.mouse.get_pos()[1] - self._scroll_drag_my0
        delta = int(round((dy / span) * max_scroll))
        self.moves_scroll = min(max(self._scroll_drag_s0 + delta, 0), max_scroll)

    def handle_moves_scroll_wheel(self, event: pygame.event.Event) -> None:
        if not hasattr(self, '_moves_list_clip_rect') or self._moves_list_clip_rect.height <= 0:
            return
        wheel_rect = self._moves_list_clip_rect.union(self._scrollbar_track_rect)
        if not wheel_rect.collidepoint(pygame.mouse.get_pos()):
            return
        delta = getattr(event, 'y', 0)
        max_scroll = max(0, getattr(self, '_scrollbar_max_scroll', 0))
        self.moves_scroll -= int(delta) * 2
        self.moves_scroll = min(max(self.moves_scroll, 0), max_scroll)

    def try_place_disk(self, pos: Tuple[int, int]) -> None:
        if self.is_game_over():
            return
        if self.play_vs_engine and self.is_live_view() and self.live_state().current_player != BLACK:
            return

        square = self.screen_to_square(pos)
        if square is None:
            return

        row, col = square
        self.make_move(row, col)

    def handle_mouse_down(self, pos: Tuple[int, int]) -> None:
        if self.play_vs_engine_button.collidepoint(pos) and self.engine_ready:
            self.play_vs_engine = not self.play_vs_engine
            if self.play_vs_engine:
                self.engine_enabled = False
                if self.is_live_view() and self.live_state().current_player == WHITE:
                    self._vs_engine_delay_until = pygame.time.get_ticks() + 500
            else:
                self._vs_engine_delay_until = None
            self.engine_cache_key = None
            return

        if self.engine_button.collidepoint(pos) and self.engine_ready:
            if self.play_vs_engine:
                return
            self.engine_enabled = not self.engine_enabled
            self.engine_cache_key = None
            return

        if self.back_button.collidepoint(pos) and self.current_index > 0:
            self.go_back()
            return

        if self.forward_button.collidepoint(pos) and self.current_index < len(self.history) - 1:
            self.go_forward()
            return

        if pos[0] >= BOARD_PIXELS and self.handle_moves_scrollbar_mouse_down(pos):
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
                    elif event.key == pygame.K_l:
                        self.reload_engine_weights()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse_down(event.pos)

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.moves_scroll_dragging = False

                elif hasattr(pygame, 'MOUSEWHEEL') and event.type == pygame.MOUSEWHEEL:
                    self.handle_moves_scroll_wheel(event)

            self.update_moves_scroll_drag()

            self.maybe_play_engine_move()

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
