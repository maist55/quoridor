import math
from collections.abc import Iterator
from time import perf_counter

from actions_quoridor import Orientation, Wall
from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction


class MyPlayer(PlayerQuoridor):
    """Joueur utilisant minimax avec élagage alpha-bêta."""

    MAX_DEPTH = 3
    MAX_TIME_PER_MOVE = 5.0

    def __init__(self, piece_type: str, goal_row: int = 0, name: str = "bob", *args, **kwargs) -> None:
        super().__init__(piece_type, goal_row, name, *args, **kwargs)

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15 * 60, **kwargs) -> Action:
        """Conserve le meilleur coup de la dernière profondeur terminée."""
        if remaining_time <= 0:
            raise RuntimeError("Time limit exceeded.")
        if current_state.is_done():
            raise RuntimeError("Game is already finished.")

        deadline = perf_counter() + min(self.MAX_TIME_PER_MOVE, remaining_time / 30)
        best_action = next(current_state.generate_possible_stateless_actions(), None)
        if best_action is None:
            raise RuntimeError("No legal action available.")

       
        for depth in range(1, self.MAX_DEPTH + 1):
            try:
                value, action = self.minimax(
                    current_state, depth, -math.inf, math.inf,
                    is_maximizing=True, deadline=deadline,
                )
            except TimeoutError:
                break

            if action is not None:
                best_action = action
            if value == math.inf:
                break

        return best_action

    def minimax(
        self,
        current_state: GameStateQuoridor,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        *,
        deadline: float = math.inf,
    ) -> tuple[float, Action | None]:
        """Renvoie (score, action), toujours du point de vue de ce joueur.

        MAX choisit notre meilleur coup ; MIN choisit la réponse adverse
        la plus défavorable. Aux feuilles, l'action vaut None.
        """
        self._check_time(deadline)
        if depth <= 0 or current_state.is_done():
            return self.evaluate(current_state), None

        best_value = -math.inf if is_maximizing else math.inf
        best_action = None

        for action in self._legal_actions(current_state, deadline):
            new_state = current_state.apply_action(action)
            value, _ = self.minimax(
                new_state, depth - 1, alpha, beta, not is_maximizing,
                deadline=deadline,
            )

            if is_maximizing:
                if best_action is None or value > best_value:
                    best_value, best_action = value, action
                alpha = max(alpha, best_value)
            else:
                if best_action is None or value < best_value:
                    best_value, best_action = value, action
                beta = min(beta, best_value)

            if beta <= alpha:
                break

        if best_action is None:
            return self.evaluate(current_state), None
        return best_value, best_action

    def evaluate(self, state: GameStateQuoridor) -> float:
        """Favorise un chemin court vers notre but et un long chemin adverse."""
        player = next(p for p in state.players if p.get_id() == self.get_id())
        opponent = next(p for p in state.players if p.get_id() != self.get_id())

        if state.scores[player.get_id()] == 1.0:
            return math.inf
        if state.scores[opponent.get_id()] == 1.0:
            return -math.inf

        
        own_distance = state._shortest_path(player)
        opponent_distance = state._shortest_path(opponent)
        if own_distance is None:
            return -math.inf
        if opponent_distance is None:
            return math.inf

        wall_advantage = (
            state.rep.remaining_walls[player.get_id()]
            - state.rep.remaining_walls[opponent.get_id()]
        )
        return opponent_distance - own_distance + 0.1 * wall_advantage

    def _legal_actions(self, state: GameStateQuoridor, deadline: float) -> Iterator[StatelessAction]:
        """Énumère tous les coups légaux en vérifiant régulièrement le temps."""
        # Essayer d'abord les déplacements vers le but aide l'élagage.
        moves = sorted(
            state._legal_moves(),
            key=lambda action: abs(action.data["destination"][0] - state.active_player.get_goal_row()),
        )
        for action in moves:
            self._check_time(deadline)
            yield action

        if state.rep.remaining_walls[state.active_player.get_id()] <= 0:
            return

        
        for row in range(state.rep.dimension):
            for col in range(state.rep.dimension):
                for orientation in Orientation:
                    self._check_time(deadline)
                    wall = Wall(row, col, orientation)
                    if state._is_wall_legal(wall):
                        yield StatelessAction({
                            "type": "horizontal" if orientation == Orientation.HORIZONTAL else "vertical",
                            "destination": (row, col),
                        })

    @staticmethod
    def _check_time(deadline: float) -> None:
        if perf_counter() >= deadline:
            raise TimeoutError("Search time budget exhausted.")
