"""Pygame chess GUI with move list and navigation."""

# Pylint doesn't understand many dynamic pygame attributes/constants.
# pylint: disable=no-member,missing-function-docstring

import os
from typing import Dict, List, Optional, Tuple

import pygame
import chess


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

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (246, 246, 105)
MOVE_DOT = (80, 80, 80)
MOVE_RING = (60, 60, 60)
BG = (26, 26, 26)

PANEL_BG = (38, 40, 44)
PANEL_BORDER = (68, 72, 78)
TEXT = (235, 235, 235)
SUBTEXT = (180, 180, 180)
ACCENT = (86, 156, 214)
ACCENT_DARK = (58, 112, 168)
BUTTON_DISABLED = (80, 80, 80)
ROW_HIGHLIGHT = (60, 92, 126)


# ============================================================
# Piece image mapping
# ============================================================
IMAGE_FILENAMES = {
    (chess.WHITE, chess.PAWN): "white-pawn.png",
    (chess.WHITE, chess.KNIGHT): "white-knight.png",
    (chess.WHITE, chess.BISHOP): "white-bishop.png",
    (chess.WHITE, chess.ROOK): "white-rook.png",
    (chess.WHITE, chess.QUEEN): "white-queen.png",
    (chess.WHITE, chess.KING): "white-king.png",
    (chess.BLACK, chess.PAWN): "black-pawn.png",
    (chess.BLACK, chess.KNIGHT): "black-knight.png",
    (chess.BLACK, chess.BISHOP): "black-bishop.png",
    (chess.BLACK, chess.ROOK): "black-rook.png",
    (chess.BLACK, chess.QUEEN): "black-queen.png",
    (chess.BLACK, chess.KING): "black-king.png",
}


def get_base_dir() -> str:
    """Return the directory containing this script (or CWD in REPL)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def candidate_image_dirs() -> List[str]:
    """Return folders that may contain piece images."""
    base = get_base_dir()
    return [
        base,
        os.path.join(base, "assets"),
    ]


class ChessGUI:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Interactive chess GUI using `python-chess` for rules and pygame for rendering."""
    def __init__(self) -> None:
        pygame.display.init()
        pygame.font.init()

        pygame.display.set_caption("2D Chess Board")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.board = chess.Board()
        self.move_history: List[chess.Move] = []
        self.current_index = 0

        self.images = self.load_piece_images()

        self.font_small = pygame.font.SysFont("arial", 18)
        self.font_body = pygame.font.SysFont("arial", 20)
        self.font_title = pygame.font.SysFont("arial", 26, bold=True)
        self.font_status = pygame.font.SysFont("arial", 22, bold=True)
        self.font_fallback = pygame.font.SysFont("arial", 28, bold=True)

        self.dragging = False
        self.drag_from: Optional[int] = None
        self.drag_piece: Optional[chess.Piece] = None
        self.drag_pos = (0, 0)
        self.legal_targets: List[int] = []

        panel_x = BOARD_PIXELS
        margin = 16
        btn_w = (PANEL_WIDTH - 3 * margin) // 2
        btn_h = 42
        btn_y = WINDOW_HEIGHT - btn_h - margin

        self.back_button = pygame.Rect(panel_x + margin, btn_y, btn_w, btn_h)
        self.forward_button = pygame.Rect(panel_x + 2 * margin + btn_w, btn_y, btn_w, btn_h)

    # --------------------------------------------------------
    # Core helpers
    # --------------------------------------------------------
    def rebuild_board(self) -> None:
        self.board = chess.Board()
        for mv in self.move_history[:self.current_index]:
            self.board.push(mv)

    def close(self) -> None:
        self.stop_drag()
        pygame.quit()

    def square_to_screen(self, square: int) -> Tuple[int, int]:
        file_ = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file_ * SQ_SIZE
        y = (7 - rank) * SQ_SIZE
        return x, y

    def screen_to_square(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        if not (0 <= x < BOARD_PIXELS and 0 <= y < BOARD_PIXELS):
            return None
        file_ = x // SQ_SIZE
        rank = 7 - (y // SQ_SIZE)
        return chess.square(file_, rank)

    def legal_moves_from(self, square: int) -> List[chess.Move]:
        return [m for m in self.board.legal_moves if m.from_square == square]

    def make_move(self, move: chess.Move) -> None:
        if self.current_index < len(self.move_history):
            self.move_history = self.move_history[:self.current_index]
        self.move_history.append(move)
        self.current_index += 1
        self.rebuild_board()
        self.stop_drag()

    def go_back(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.rebuild_board()
            self.stop_drag()

    def go_forward(self) -> None:
        if self.current_index < len(self.move_history):
            self.current_index += 1
            self.rebuild_board()
            self.stop_drag()

    def reset_game(self) -> None:
        self.move_history = []
        self.current_index = 0
        self.rebuild_board()
        self.stop_drag()

    # --------------------------------------------------------
    # Image loading
    # --------------------------------------------------------
    def load_piece_images(self) -> Dict[Tuple[bool, int], pygame.Surface]:
        images: Dict[Tuple[bool, int], pygame.Surface] = {}

        for piece_key, filename in IMAGE_FILENAMES.items():
            loaded = False

            for folder in candidate_image_dirs():
                path = os.path.join(folder, filename)
                if os.path.exists(path):
                    img = pygame.image.load(path).convert_alpha()
                    img = pygame.transform.smoothscale(img, (SQ_SIZE - 10, SQ_SIZE - 10))
                    images[piece_key] = img
                    loaded = True
                    break

            if not loaded:
                print(f"Warning: could not find image file '{filename}'")

        return images

    # --------------------------------------------------------
    # SAN move list
    # --------------------------------------------------------
    def get_san_rows(self):
        temp_board = chess.Board()
        san_moves: List[str] = []

        for mv in self.move_history:
            san_moves.append(temp_board.san(mv))
            temp_board.push(mv)

        rows = []
        i = 0
        while i < len(san_moves):
            move_no = i // 2 + 1
            white_san = san_moves[i]
            white_idx = i + 1

            if i + 1 < len(san_moves):
                black_san = san_moves[i + 1]
                black_idx = i + 2
            else:
                black_san = ""
                black_idx = None

            rows.append((move_no, white_san, black_san, white_idx, black_idx))
            i += 2

        return rows

    # --------------------------------------------------------
    # Board drawing
    # --------------------------------------------------------
    def draw_board(self) -> None:
        self.screen.fill(BG)

        for rank in range(8):
            for file_ in range(8):
                color = LIGHT if (rank + file_) % 2 == 0 else DARK
                x = file_ * SQ_SIZE
                y = rank * SQ_SIZE
                pygame.draw.rect(self.screen, color, (x, y, SQ_SIZE, SQ_SIZE))

        if self.drag_from is not None:
            x, y = self.square_to_screen(self.drag_from)
            pygame.draw.rect(self.screen, HIGHLIGHT, (x, y, SQ_SIZE, SQ_SIZE), width=5)

        self.draw_coordinates()
        self.draw_legal_targets()

    def draw_coordinates(self) -> None:
        for file_ in range(8):
            label = self.font_small.render(chr(ord("a") + file_), True, (45, 45, 45))
            self.screen.blit(label, (file_ * SQ_SIZE + SQ_SIZE - 16, BOARD_PIXELS - 20))

        for rank in range(8):
            label = self.font_small.render(str(8 - rank), True, (45, 45, 45))
            self.screen.blit(label, (4, rank * SQ_SIZE + 4))

    def draw_legal_targets(self) -> None:
        for sq in self.legal_targets:
            x, y = self.square_to_screen(sq)
            target_piece = self.board.piece_at(sq)
            cx = x + SQ_SIZE // 2
            cy = y + SQ_SIZE // 2
            if target_piece is None:
                pygame.draw.circle(self.screen, MOVE_DOT, (cx, cy), 10)
            else:
                pygame.draw.circle(self.screen, MOVE_RING, (cx, cy), SQ_SIZE // 2 - 8, width=5)

    def draw_piece_on_square(self, piece: chess.Piece, square: int) -> None:
        x, y = self.square_to_screen(square)
        center = (x + SQ_SIZE // 2, y + SQ_SIZE // 2)

        key = (piece.color, piece.piece_type)
        if key in self.images:
            rect = self.images[key].get_rect(center=center)
            self.screen.blit(self.images[key], rect)
        else:
            self.draw_fallback_piece(piece, center)

    def draw_fallback_piece(self, piece: chess.Piece, center: Tuple[int, int]) -> None:
        cx, cy = center
        radius = SQ_SIZE // 2 - 10

        if piece.color == chess.WHITE:
            fill = (245, 245, 245)
            border = (30, 30, 30)
            text_color = (20, 20, 20)
        else:
            fill = (35, 35, 35)
            border = (230, 230, 230)
            text_color = (245, 245, 245)

        piece_letter = {
            chess.PAWN: "P",
            chess.KNIGHT: "N",
            chess.BISHOP: "B",
            chess.ROOK: "R",
            chess.QUEEN: "Q",
            chess.KING: "K",
        }[piece.piece_type]

        pygame.draw.circle(self.screen, fill, center, radius)
        pygame.draw.circle(self.screen, border, center, radius, width=2)

        text = self.font_fallback.render(piece_letter, True, text_color)
        rect = text.get_rect(center=(cx, cy))
        self.screen.blit(text, rect)

    def draw_pieces(self) -> None:
        for square, piece in self.board.piece_map().items():
            if self.dragging and self.drag_from == square:
                continue
            self.draw_piece_on_square(piece, square)

        if self.dragging and self.drag_piece is not None:
            self.draw_dragging_piece()

    def draw_dragging_piece(self) -> None:
        assert self.drag_piece is not None

        key = (self.drag_piece.color, self.drag_piece.piece_type)
        mx, my = self.drag_pos

        if key in self.images:
            rect = self.images[key].get_rect(center=(mx, my))
            self.screen.blit(self.images[key], rect)
        else:
            self.draw_fallback_piece(self.drag_piece, (mx, my))

    def draw_game_over(self) -> None:
        if not self.board.is_game_over():
            return

        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            msg = f"Checkmate - {winner} wins"
        elif self.board.is_stalemate():
            msg = "Draw - Stalemate"
        elif self.board.is_insufficient_material():
            msg = "Draw - Insufficient material"
        elif self.board.can_claim_threefold_repetition():
            msg = "Draw - Threefold repetition"
        else:
            msg = "Game over"

        overlay = pygame.Surface((BOARD_PIXELS, BOARD_PIXELS), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        self.screen.blit(overlay, (0, 0))

        text = self.font_status.render(msg, True, (255, 255, 255))
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
        panel_rect = pygame.Rect(BOARD_PIXELS, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(
            self.screen, PANEL_BORDER, (BOARD_PIXELS, 0), (BOARD_PIXELS, WINDOW_HEIGHT), 2
        )

        panel_x = BOARD_PIXELS + 16
        y = 16

        title = self.font_title.render("Moves", True, TEXT)
        self.screen.blit(title, (panel_x, y))
        y += 40

        if self.current_index == 0:
            status_text = "Start position"
        elif self.current_index == len(self.move_history):
            status_text = f"Live position ({self.current_index} ply)"
        else:
            status_text = f"History view ({self.current_index}/{len(self.move_history)} ply)"

        status = self.font_body.render(status_text, True, SUBTEXT)
        self.screen.blit(status, (panel_x, y))
        y += 34

        turn_text = "White to move" if self.board.turn == chess.WHITE else "Black to move"
        turn_surface = self.font_body.render(turn_text, True, TEXT)
        self.screen.blit(turn_surface, (panel_x, y))
        y += 18

        pygame.draw.line(
            self.screen,
            PANEL_BORDER,
            (BOARD_PIXELS + 12, y + 10),
            (WINDOW_WIDTH - 12, y + 10),
            1,
        )
        y += 24

        rows = self.get_san_rows()
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
            move_no, white_san, black_san, white_idx, black_idx = rows[row_i]
            row_y = y + (row_i - start_row) * row_height

            if self.current_index in (white_idx, black_idx):
                highlight_rect = pygame.Rect(
                    BOARD_PIXELS + 10, row_y - 2, PANEL_WIDTH - 20, row_height - 2
                )
                pygame.draw.rect(self.screen, ROW_HIGHLIGHT, highlight_rect, border_radius=6)

            num_txt = self.font_small.render(f"{move_no}.", True, SUBTEXT)
            self.screen.blit(num_txt, (col1, row_y + 4))

            white_color = TEXT if white_idx == self.current_index else (220, 220, 220)
            black_color = TEXT if black_idx == self.current_index else (220, 220, 220)

            white_txt = self.font_body.render(white_san, True, white_color)
            self.screen.blit(white_txt, (col2, row_y + 2))

            if black_san:
                black_txt = self.font_body.render(black_san, True, black_color)
                self.screen.blit(black_txt, (col3, row_y + 2))

        self.draw_button(self.back_button, "Back", self.current_index > 0)
        self.draw_button(
            self.forward_button, "Forward", self.current_index < len(self.move_history)
        )

        help_text = self.font_small.render("Left/Right arrows also work", True, SUBTEXT)
        help_rect = help_text.get_rect(
            midbottom=(BOARD_PIXELS + PANEL_WIDTH // 2, self.back_button.top - 10)
        )
        self.screen.blit(help_text, help_rect)

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------
    def start_drag(self, pos: Tuple[int, int]) -> None:
        if self.board.is_game_over():
            return

        sq = self.screen_to_square(pos)
        if sq is None:
            return

        piece = self.board.piece_at(sq)
        if piece is None or piece.color != self.board.turn:
            return

        self.dragging = True
        self.drag_from = sq
        self.drag_piece = piece
        self.drag_pos = pos
        self.legal_targets = [m.to_square for m in self.legal_moves_from(sq)]

    def stop_drag(self) -> None:
        self.dragging = False
        self.drag_from = None
        self.drag_piece = None
        self.legal_targets = []

    def try_drop(self, pos: Tuple[int, int]) -> None:
        if not self.dragging or self.drag_from is None or self.drag_piece is None:
            return

        to_sq = self.screen_to_square(pos)
        if to_sq is None:
            self.stop_drag()
            return

        legal_moves = self.legal_moves_from(self.drag_from)
        chosen_move = None

        for move in legal_moves:
            if move.to_square == to_sq:
                if move.promotion is not None:
                    if move.promotion == chess.QUEEN:
                        chosen_move = move
                        break
                else:
                    chosen_move = move
                    break

        if chosen_move is not None:
            self.make_move(chosen_move)
        else:
            self.stop_drag()

    def handle_mouse_down(self, pos: Tuple[int, int]) -> None:
        if self.back_button.collidepoint(pos) and self.current_index > 0:
            self.go_back()
            return

        if self.forward_button.collidepoint(pos) and self.current_index < len(self.move_history):
            self.go_forward()
            return

        if pos[0] < BOARD_PIXELS:
            self.start_drag(pos)

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

                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.drag_pos = event.pos

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.dragging:
                        self.try_drop(event.pos)

            self.draw_board()
            self.draw_pieces()
            self.draw_game_over()
            self.draw_side_panel()
            pygame.display.flip()

        self.close()


def main() -> None:
    """Run the chess GUI application."""
    ChessGUI().run()


if __name__ == "__main__":
    main()
    