"""Neural-network + MCTS Reversi engine with self-play training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0, logits, -1e9)
    max_logits = np.max(masked, axis=1, keepdims=True)
    exps = np.exp(masked - max_logits)
    exps *= mask
    sums = np.sum(exps, axis=1, keepdims=True)
    sums = np.maximum(sums, 1e-8)
    return exps / sums


def _conv2d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int = 1) -> Tuple[np.ndarray, List[np.ndarray]]:
    """NCHW conv, kernel 3x3. Returns (out, list of region tensors per spatial cell)."""
    bsz, _, h, w_in = x.shape
    cout, cin, kh, kw = w.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((bsz, cout, h, w_in), dtype=np.float32)
    regions: List[np.ndarray] = []
    for yy in range(h):
        for xx in range(w_in):
            reg = x_pad[:, :, yy : yy + kh, xx : xx + kw]
            regions.append(reg)
            out[:, :, yy, xx] = np.einsum('bcij,fcij->bf', reg, w) + b
    return out, regions


def _conv2d_backward(
    d_out: np.ndarray,
    w: np.ndarray,
    regions: List[np.ndarray],
    x_shape: Tuple[int, int, int, int],
    pad: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradients w.r.t. W, b, and input x (NCHW)."""
    bsz, cin, h, w_in = x_shape
    cout, _, kh, kw = w.shape
    dw = np.zeros_like(w, dtype=np.float32)
    db = np.zeros(cout, dtype=np.float32)
    dx_pad = np.zeros((bsz, cin, h + 2 * pad, w_in + 2 * pad), dtype=np.float32)
    idx = 0
    for yy in range(h):
        for xx in range(w_in):
            reg = regions[idx]
            idx += 1
            dout_hw = d_out[:, :, yy, xx]
            db += np.sum(dout_hw, axis=0)
            dw += np.einsum('bf,bcij->fcij', dout_hw, reg)
            d_reg = np.einsum('bf,fcij->bcij', dout_hw, w)
            dx_pad[:, :, yy : yy + kh, xx : xx + kw] += d_reg
    dx = dx_pad[:, :, pad : pad + h, pad : pad + w_in]
    return dw, db, dx


class PolicyValueNet:
    """Policy + value network: CNN (AlphaZero-style) or legacy MLP."""

    arch: str

    def __init__(
        self,
        hidden_size: int = 128,
        learning_rate: float = 3e-3,
        seed: int = 0,
        arch: str = 'cnn',
        conv_channels: int = 64,
    ) -> None:
        self.arch = arch
        self.lr = float(learning_rate)
        rng = np.random.default_rng(seed)
        if arch == 'mlp':
            input_dim = 3 * BOARD_SIZE * BOARD_SIZE
            self.hidden_size = hidden_size
            self.w1 = rng.normal(0.0, 0.05, size=(input_dim, hidden_size)).astype(np.float32)
            self.b1 = np.zeros(hidden_size, dtype=np.float32)
            self.wp = rng.normal(0.0, 0.05, size=(hidden_size, ACTION_SIZE)).astype(np.float32)
            self.bp = np.zeros(ACTION_SIZE, dtype=np.float32)
            self.wv = rng.normal(0.0, 0.05, size=(hidden_size, 1)).astype(np.float32)
            self.bv = np.zeros(1, dtype=np.float32)
        elif arch == 'cnn':
            self.conv_c = conv_channels
            c = conv_channels
            self.w_conv = rng.normal(0.0, 0.05, size=(c, 3, 3, 3)).astype(np.float32)
            self.b_conv = np.zeros(c, dtype=np.float32)
            self.w_pol = rng.normal(0.0, 0.05, size=(c, ACTION_SIZE)).astype(np.float32)
            self.b_pol = np.zeros(ACTION_SIZE, dtype=np.float32)
            self.w_val = rng.normal(0.0, 0.05, size=(c, 1)).astype(np.float32)
            self.b_val = np.zeros(1, dtype=np.float32)
        else:
            raise ValueError(f"Unknown arch {arch!r}, expected 'cnn' or 'mlp'")

    def _planes_batch(self, states_flat: np.ndarray) -> np.ndarray:
        return states_flat.reshape(states_flat.shape[0], 3, BOARD_SIZE, BOARD_SIZE).astype(np.float32)

    def forward(self, states_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.arch == 'mlp':
            h = np.tanh(states_flat @ self.w1 + self.b1)
            logits = h @ self.wp + self.bp
            value = np.tanh((h @ self.wv + self.bv).squeeze(-1))
            return logits, value
        x = self._planes_batch(states_flat)
        conv_pre, regions = _conv2d_forward(x, self.w_conv, self.b_conv, pad=1)
        self._last_conv_regions = regions
        self._last_conv_pre = conv_pre
        h_relu = np.maximum(conv_pre, 0.0)
        gap = np.mean(h_relu, axis=(2, 3))
        logits = gap @ self.w_pol + self.b_pol
        value_pre = (gap @ self.w_val + self.b_val).squeeze(-1)
        value = np.tanh(value_pre)
        self._last_gap = gap
        self._last_h_relu = h_relu
        self._last_value_pre = value_pre
        self._last_x = x
        return logits, value

    def predict(self, state: ReversiState) -> Tuple[np.ndarray, float]:
        x = encode_state(state)[None, :]
        logits, value = self.forward(x)
        mask = valid_action_mask(state)[None, :]
        policy = _softmax_masked(logits, mask)[0]
        return policy, float(value[0])

    def train_batch(
        self,
        states: np.ndarray,
        target_policy: np.ndarray,
        target_value: np.ndarray,
        valid_mask: np.ndarray,
    ) -> float:
        if self.arch == 'mlp':
            return self._train_batch_mlp(states, target_policy, target_value, valid_mask)
        return self._train_batch_cnn(states, target_policy, target_value, valid_mask)

    def _train_batch_mlp(
        self,
        states: np.ndarray,
        target_policy: np.ndarray,
        target_value: np.ndarray,
        valid_mask: np.ndarray,
    ) -> float:
        logits, values = self.forward(states)
        probs = _softmax_masked(logits, valid_mask)

        batch_size = states.shape[0]
        eps = 1e-8
        policy_loss = -np.sum(target_policy * np.log(probs + eps)) / batch_size
        value_loss = np.mean((values - target_value) ** 2)
        loss = policy_loss + value_loss

        grad_logits = (probs - target_policy) / batch_size
        grad_logits *= valid_mask

        grad_value = 2.0 * (values - target_value) / batch_size
        grad_value *= 1.0 - values**2

        h = np.tanh(states @ self.w1 + self.b1)
        grad_wp = h.T @ grad_logits
        grad_bp = np.sum(grad_logits, axis=0)

        grad_wv = h.T @ grad_value[:, None]
        grad_bv = np.sum(grad_value)

        grad_h = grad_logits @ self.wp.T + grad_value[:, None] @ self.wv.T
        grad_h *= 1.0 - h**2

        grad_w1 = states.T @ grad_h
        grad_b1 = np.sum(grad_h, axis=0)

        self.wp -= self.lr * grad_wp
        self.bp -= self.lr * grad_bp
        self.wv -= self.lr * grad_wv
        self.bv -= self.lr * np.array([grad_bv], dtype=np.float32)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

        return float(loss)

    def _train_batch_cnn(
        self,
        states: np.ndarray,
        target_policy: np.ndarray,
        target_value: np.ndarray,
        valid_mask: np.ndarray,
    ) -> float:
        logits, values = self.forward(states)
        probs = _softmax_masked(logits, valid_mask)

        batch_size = states.shape[0]
        eps = 1e-8
        policy_loss = -np.sum(target_policy * np.log(probs + eps)) / batch_size
        value_loss = np.mean((values - target_value) ** 2)
        loss = policy_loss + value_loss

        grad_logits = (probs - target_policy) / batch_size
        grad_logits *= valid_mask

        grad_value = 2.0 * (values - target_value) / batch_size
        grad_value *= 1.0 - values**2

        gap = self._last_gap
        h_relu = self._last_h_relu
        conv_pre = self._last_conv_pre
        regions = self._last_conv_regions
        x = self._last_x
        value_pre = self._last_value_pre

        d_gap = grad_logits @ self.w_pol.T + grad_value[:, None] @ self.w_val.T
        hw = BOARD_SIZE * BOARD_SIZE
        d_h = d_gap[:, :, None, None] / float(hw)

        d_conv_pre = d_h * (conv_pre > 0).astype(np.float32)

        grad_w_pol = gap.T @ grad_logits
        grad_b_pol = np.sum(grad_logits, axis=0)
        grad_w_val = gap.T @ grad_value[:, None]
        grad_b_val = np.sum(grad_value)

        dw_conv, db_conv, _dx = _conv2d_backward(
            d_conv_pre, self.w_conv, regions, x.shape, pad=1
        )

        self.w_pol -= self.lr * grad_w_pol
        self.b_pol -= self.lr * grad_b_pol
        self.w_val -= self.lr * grad_w_val
        self.b_val -= self.lr * np.array([grad_b_val], dtype=np.float32)
        self.w_conv -= self.lr * dw_conv
        self.b_conv -= self.lr * db_conv

        return float(loss)

    def save_dict(self) -> Dict[str, np.ndarray]:
        if self.arch == 'mlp':
            return {
                'arch': np.array('mlp'),
                'w1': self.w1,
                'b1': self.b1,
                'wp': self.wp,
                'bp': self.bp,
                'wv': self.wv,
                'bv': self.bv,
            }
        return {
            'arch': np.array('cnn'),
            'w_conv': self.w_conv,
            'b_conv': self.b_conv,
            'w_pol': self.w_pol,
            'b_pol': self.b_pol,
            'w_val': self.w_val,
            'b_val': self.b_val,
        }

    @classmethod
    def from_npz(cls, data: Any) -> PolicyValueNet:
        arch = str(data['arch'].item()) if 'arch' in data.files else ''
        if arch == 'cnn' or 'w_conv' in data.files:
            w_conv = np.asarray(data['w_conv'], dtype=np.float32)
            c = int(w_conv.shape[0])
            net = cls(arch='cnn', conv_channels=c, learning_rate=3e-3, seed=0)
            net.w_conv = w_conv
            net.b_conv = np.asarray(data['b_conv'], dtype=np.float32)
            net.w_pol = np.asarray(data['w_pol'], dtype=np.float32)
            net.b_pol = np.asarray(data['b_pol'], dtype=np.float32)
            net.w_val = np.asarray(data['w_val'], dtype=np.float32)
            net.b_val = np.asarray(data['b_val'], dtype=np.float32)
            return net
        w1 = np.asarray(data['w1'], dtype=np.float32)
        hidden_size = int(w1.shape[1])
        net = cls(hidden_size=hidden_size, arch='mlp', learning_rate=1e-2, seed=0)
        net.w1 = w1
        net.b1 = np.asarray(data['b1'], dtype=np.float32)
        net.wp = np.asarray(data['wp'], dtype=np.float32)
        net.bp = np.asarray(data['bp'], dtype=np.float32)
        net.wv = np.asarray(data['wv'], dtype=np.float32)
        net.bv = np.asarray(data['bv'], dtype=np.float32)
        return net


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


def _apply_root_dirichlet_noise(
    root: MCTSNode,
    mask: np.ndarray,
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> None:
    """AlphaZero-style exploration: blend uniform Dirichlet noise into root priors (legal moves only)."""
    legal = np.where(mask > 0)[0]
    if len(legal) == 0:
        return
    noise = rng.dirichlet(np.full(len(legal), alpha, dtype=np.float64)).astype(np.float32)
    for i, action in enumerate(legal):
        child = root.children.get(int(action))
        if child is None:
            continue
        child.prior = (1.0 - epsilon) * child.prior + epsilon * float(noise[i])


class MCTS:
    def __init__(self, net: PolicyValueNet, simulations: int = 128, c_puct: float = 1.4) -> None:
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = np.random.default_rng(0)

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

    def _run_search(
        self,
        root_state: ReversiState,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.35,
        dirichlet_epsilon: float = 0.25,
        simulations: Optional[int] = None,
    ) -> MCTSNode:
        sims = self.simulations if simulations is None else max(1, int(simulations))
        root = MCTSNode(root_state)
        self._expand(root)
        if add_root_noise and root.children:
            _apply_root_dirichlet_noise(
                root,
                valid_action_mask(root_state),
                dirichlet_alpha,
                dirichlet_epsilon,
                self.rng,
            )

        for _ in range(sims):
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

        return root

    def evaluate(self, root_state: ReversiState, simulations: Optional[int] = None) -> float:
        """Mean root value after search, from root_state.current_player's perspective ([-1, 1])."""
        root = self._run_search(root_state, add_root_noise=False, simulations=simulations)
        return root.value

    def run(
        self,
        root_state: ReversiState,
        temperature: float = 1.0,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.35,
        dirichlet_epsilon: float = 0.25,
    ) -> np.ndarray:
        root = self._run_search(
            root_state,
            add_root_noise=add_root_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

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
    """AlphaZero loop: MCTS self-play (root Dirichlet + visit policy) then SGD on (s, π, z)."""

    def __init__(self, net: PolicyValueNet, simulations: int = 48, seed: int = 0) -> None:
        self.net = net
        self.mcts = MCTS(net, simulations=simulations)
        self.mcts.rng = np.random.default_rng(seed)

    def self_play_game(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = initial_state()
        game_states: List[np.ndarray] = []
        game_masks: List[np.ndarray] = []
        game_policies: List[np.ndarray] = []
        players: List[int] = []

        move_index = 0
        while not is_terminal(state):
            # AlphaZero-style: high temperature for opening, anneal later; root noise only in self-play.
            temperature = 1.0 if move_index < 24 else (0.25 if move_index < 48 else 1e-6)
            policy = self.mcts.run(
                state,
                temperature=temperature,
                add_root_noise=True,
                dirichlet_alpha=0.35,
                dirichlet_epsilon=0.25,
            )

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

    def train(
        self,
        iterations: int = 12,
        games_per_iteration: int = 4,
        batch_size: int = 64,
        lr_decay: float = 0.985,
        max_replay_samples: int = 50_000,
        epochs_per_iteration: int = 1,
    ) -> None:
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

            n = states_all.shape[0]
            if n > max_replay_samples:
                pick = np.random.choice(n, size=max_replay_samples, replace=False)
                states_all = states_all[pick]
                masks_all = masks_all[pick]
                policies_all = policies_all[pick]
                values_all = values_all[pick]
                n = max_replay_samples

            if it > 1:
                self.net.lr *= lr_decay

            epoch_loss = 0.0
            batches = 0
            for _epoch in range(epochs_per_iteration):
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    batch_idx = idx[start:start + batch_size]
                    loss = self.net.train_batch(
                        states_all[batch_idx],
                        policies_all[batch_idx],
                        values_all[batch_idx],
                        masks_all[batch_idx],
                    )
                    epoch_loss += loss
                    batches += 1

            print(
                f"Iteration {it:02d} | samples={n:5d} | lr={self.net.lr:.5f} | "
                f"loss={epoch_loss / max(1, batches):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    print(
        "\n=== Self-play TRAINING (this file) — not the pygame app. "
        "To play: python reversi.py ===\n",
        flush=True,
    )
    # CNN + PUCT MCTS self-play. Defaults favor CPU/laptop runs (fewer sims, smaller net,
    # fewer games per round). Raise simulations / conv_channels / iterations for stronger play.
    network = PolicyValueNet(arch='cnn', conv_channels=32, learning_rate=0.003, seed=42)
    trainer = SelfPlayTrainer(network, simulations=48, seed=42)
    trainer.train(
        iterations=10,
        games_per_iteration=4,
        batch_size=64,
        lr_decay=0.988,
        max_replay_samples=50_000,
        epochs_per_iteration=1,
    )
    weights_file = Path(__file__).with_name('reversi_engine_weights.npz')
    np.savez(weights_file, **network.save_dict())
    print(f'Saved trained weights to {weights_file}')
