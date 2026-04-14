"""Neural-network + MCTS Reversi engine with self-play training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BOARD_SIZE = 8
EMPTY = 0
BLACK = 1
WHITE = -1
PASS_MOVE = BOARD_SIZE * BOARD_SIZE
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


@dataclass(frozen=True)
class ReversiState:
    board: np.ndarray  # shape: (8, 8), values in {-1, 0, 1}
    current_player: int


def initial_state() -> ReversiState:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[3, 3] = WHITE
    board[3, 4] = BLACK
    board[4, 3] = BLACK
    board[4, 4] = WHITE
    return ReversiState(board=board, current_player=BLACK)


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def legal_moves(state: ReversiState, player: Optional[int] = None) -> Dict[int, List[Tuple[int, int]]]:
    board = state.board
    turn = state.current_player if player is None else player
    moves: Dict[int, List[Tuple[int, int]]] = {}

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row, col] != EMPTY:
                continue

            flips: List[Tuple[int, int]] = []
            for dr, dc in DIRECTIONS:
                r, c = row + dr, col + dc
                line: List[Tuple[int, int]] = []
                while in_bounds(r, c) and board[r, c] == -turn:
                    line.append((r, c))
                    r += dr
                    c += dc
                if line and in_bounds(r, c) and board[r, c] == turn:
                    flips.extend(line)

            if flips:
                action = row * BOARD_SIZE + col
                moves[action] = flips

    return moves


def apply_action(state: ReversiState, action: int) -> ReversiState:
    board = state.board.copy()
    turn = state.current_player
    moves = legal_moves(state, turn)

    if action == PASS_MOVE:
        return ReversiState(board=board, current_player=-turn)

    flips = moves.get(action)
    if flips is None:
        raise ValueError(f"Illegal action {action} for player {turn}")

    row, col = divmod(action, BOARD_SIZE)
    board[row, col] = turn
    for flip_r, flip_c in flips:
        board[flip_r, flip_c] = turn

    return ReversiState(board=board, current_player=-turn)


def is_terminal(state: ReversiState) -> bool:
    return not legal_moves(state, BLACK) and not legal_moves(state, WHITE)


def winner(state: ReversiState) -> int:
    black = int(np.sum(state.board == BLACK))
    white = int(np.sum(state.board == WHITE))
    if black > white:
        return BLACK
    if white > black:
        return WHITE
    return EMPTY


def valid_action_mask(state: ReversiState) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    moves = legal_moves(state)
    if not moves:
        mask[PASS_MOVE] = 1.0
        return mask
    for action in moves:
        mask[action] = 1.0
    return mask


def encode_state(state: ReversiState) -> np.ndarray:
    player = state.current_player
    own = (state.board == player).astype(np.float32)
    opp = (state.board == -player).astype(np.float32)
    side_to_move = np.full((BOARD_SIZE, BOARD_SIZE), 1.0 if player == BLACK else 0.0, dtype=np.float32)
    stacked = np.stack([own, opp, side_to_move], axis=0)
    return stacked.reshape(-1)


class PolicyValueNet:
    """Small MLP producing policy logits and value."""

    def __init__(self, hidden_size: int = 128, learning_rate: float = 1e-2, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        input_dim = 3 * BOARD_SIZE * BOARD_SIZE
        self.lr = learning_rate
        self.w1 = rng.normal(0.0, 0.05, size=(input_dim, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.wp = rng.normal(0.0, 0.05, size=(hidden_size, ACTION_SIZE)).astype(np.float32)
        self.bp = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.wv = rng.normal(0.0, 0.05, size=(hidden_size, 1)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1 + self.b1)
        logits = h @ self.wp + self.bp
        value = np.tanh((h @ self.wv + self.bv).squeeze(-1))
        return logits, value

    @staticmethod
    def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.where(mask > 0, logits, -1e9)
        max_logits = np.max(masked, axis=1, keepdims=True)
        exps = np.exp(masked - max_logits)
        exps *= mask
        sums = np.sum(exps, axis=1, keepdims=True)
        sums = np.maximum(sums, 1e-8)
        return exps / sums

    def predict(self, state: ReversiState) -> Tuple[np.ndarray, float]:
        x = encode_state(state)[None, :]
        logits, value = self.forward(x)
        mask = valid_action_mask(state)[None, :]
        policy = self._softmax_masked(logits, mask)[0]
        return policy, float(value[0])

    def train_batch(
        self,
        states: np.ndarray,
        target_policy: np.ndarray,
        target_value: np.ndarray,
        valid_mask: np.ndarray,
    ) -> float:
        logits, values = self.forward(states)
        probs = self._softmax_masked(logits, valid_mask)

        batch_size = states.shape[0]
        eps = 1e-8
        policy_loss = -np.sum(target_policy * np.log(probs + eps)) / batch_size
        value_loss = np.mean((values - target_value) ** 2)
        loss = policy_loss + value_loss

        grad_logits = (probs - target_policy) / batch_size
        grad_logits *= valid_mask

        grad_value = 2.0 * (values - target_value) / batch_size
        grad_value *= (1.0 - values**2)

        h = np.tanh(states @ self.w1 + self.b1)
        grad_wp = h.T @ grad_logits
        grad_bp = np.sum(grad_logits, axis=0)

        grad_wv = h.T @ grad_value[:, None]
        grad_bv = np.sum(grad_value)

        grad_h = grad_logits @ self.wp.T + grad_value[:, None] @ self.wv.T
        grad_h *= (1.0 - h**2)

        grad_w1 = states.T @ grad_h
        grad_b1 = np.sum(grad_h, axis=0)

        self.wp -= self.lr * grad_wp
        self.bp -= self.lr * grad_bp
        self.wv -= self.lr * grad_wv
        self.bv -= self.lr * np.array([grad_bv], dtype=np.float32)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

        return float(loss)


class MCTSNode:
    def __init__(self, state: ReversiState, prior: float = 0.0) -> None:
        self.state = state
        self.prior = prior
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, net: PolicyValueNet, simulations: int = 128, c_puct: float = 1.4) -> None:
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct

    def _expand(self, node: MCTSNode) -> float:
        if is_terminal(node.state):
            win = winner(node.state)
            if win == EMPTY:
                return 0.0
            return 1.0 if win == node.state.current_player else -1.0

        policy, value = self.net.predict(node.state)
        mask = valid_action_mask(node.state)
        for action in np.where(mask > 0)[0]:
            child_state = apply_action(node.state, int(action))
            node.children[int(action)] = MCTSNode(child_state, prior=float(policy[action]))
        return value

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        best_action = -1
        best_child: Optional[MCTSNode] = None
        best_score = -1e9
        total_visits = max(1, node.visit_count)

        for action, child in node.children.items():
            q = -child.value
            u = self.c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("No child selected during MCTS.")
        return best_action, best_child

    def run(self, root_state: ReversiState, temperature: float = 1.0) -> np.ndarray:
        root = MCTSNode(root_state)
        self._expand(root)

        for _ in range(self.simulations):
            node = root
            path = [node]

            while node.children:
                _, node = self._select_child(node)
                path.append(node)

            value = self._expand(node)
            for path_node in reversed(path):
                path_node.visit_count += 1
                path_node.value_sum += value
                value = -value

        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if np.sum(visits) == 0:
            visits[PASS_MOVE] = 1.0

        if temperature <= 1e-6:
            policy = np.zeros_like(visits)
            policy[int(np.argmax(visits))] = 1.0
            return policy

        visits = visits ** (1.0 / temperature)
        return visits / np.sum(visits)


class SelfPlayTrainer:
    def __init__(self, net: PolicyValueNet, simulations: int = 96) -> None:
        self.net = net
        self.mcts = MCTS(net, simulations=simulations)

    def self_play_game(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = initial_state()
        game_states: List[np.ndarray] = []
        game_masks: List[np.ndarray] = []
        game_policies: List[np.ndarray] = []
        players: List[int] = []

        move_index = 0
        while not is_terminal(state):
            temperature = 1.0 if move_index < 12 else 0.2
            policy = self.mcts.run(state, temperature=temperature)

            game_states.append(encode_state(state))
            game_masks.append(valid_action_mask(state))
            game_policies.append(policy)
            players.append(state.current_player)

            action = int(np.random.choice(np.arange(ACTION_SIZE), p=policy))
            state = apply_action(state, action)
            move_index += 1

        win = winner(state)
        outcomes = np.array(
            [0.0 if win == EMPTY else (1.0 if p == win else -1.0) for p in players], dtype=np.float32
        )
        return (
            np.stack(game_states).astype(np.float32),
            np.stack(game_masks).astype(np.float32),
            np.stack(game_policies).astype(np.float32),
            outcomes,
        )

    def train(self, iterations: int = 20, games_per_iteration: int = 8, batch_size: int = 64) -> None:
        replay_states: List[np.ndarray] = []
        replay_masks: List[np.ndarray] = []
        replay_policies: List[np.ndarray] = []
        replay_values: List[np.ndarray] = []

        for it in range(1, iterations + 1):
            for _ in range(games_per_iteration):
                states, masks, policies, values = self.self_play_game()
                replay_states.append(states)
                replay_masks.append(masks)
                replay_policies.append(policies)
                replay_values.append(values)

            states_all = np.concatenate(replay_states, axis=0)
            masks_all = np.concatenate(replay_masks, axis=0)
            policies_all = np.concatenate(replay_policies, axis=0)
            values_all = np.concatenate(replay_values, axis=0)

            idx = np.random.permutation(states_all.shape[0])
            epoch_loss = 0.0
            batches = 0

            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start:start + batch_size]
                loss = self.net.train_batch(
                    states_all[batch_idx],
                    policies_all[batch_idx],
                    values_all[batch_idx],
                    masks_all[batch_idx],
                )
                epoch_loss += loss
                batches += 1

            print(f"Iteration {it:02d} | samples={len(idx):4d} | loss={epoch_loss / max(1, batches):.4f}")


if __name__ == "__main__":
    network = PolicyValueNet(hidden_size=128, learning_rate=0.01, seed=42)
    trainer = SelfPlayTrainer(network, simulations=64)
    trainer.train(iterations=10, games_per_iteration=4, batch_size=64)
    weights_file = Path(__file__).with_name('reversi_engine_weights.npz')
    np.savez(
        weights_file,
        w1=network.w1,
        b1=network.b1,
        wp=network.wp,
        bp=network.bp,
        wv=network.wv,
        bv=network.bv,
    )
    print(f'Saved trained weights to {weights_file}')
