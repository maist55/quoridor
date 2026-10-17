# INF8175 - Projet Quoridor
# Marouane Istadad - 2346401
# Asmaa Sabouri - 2224168

import math
from collections import deque
from collections.abc import Iterator
from time import perf_counter

from actions_quoridor import Orientation, Wall
from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from seahorse.player.player import Player


class MyPlayer(PlayerQuoridor):
    """Joueur utilisant minimax avec élagage alpha-bêta.

    V2 (par rapport à la V1) :
      - Génération des murs restreinte aux positions "utiles" : seuls les
        murs qui touchent le plus court chemin d'au moins un des deux
        joueurs sont considérés, au lieu de tester systématiquement les
        ~150 positions du plateau à chaque nœud. Un mur qui ne touche
        aucun des deux chemins ne peut pas changer la distance restante
        d'un joueur, il est donc inutile de l'explorer. Cela réduit très
        fortement le facteur de branchement et permet une recherche plus
        profonde dans le même budget de temps.
      - Heuristique enrichie d'un terme de mobilité (nombre de cases
        immédiatement accessibles par chaque joueur), en plus de la
        distance au but et de l'avantage en murs restants.
    """

    MAX_DEPTH = 4
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
        """Favorise un chemin court vers notre but, un long chemin adverse,
        un avantage en murs restants et une bonne mobilité immédiate.
        """
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

        # Nombre de cases immédiatement accessibles : un joueur acculé
        # (peu d'options) est en plus mauvaise posture qu'un simple écart
        # de distance ne le laisse paraître.
        own_mobility = len(state._reachable_neighbours(state.rep.pawn_positions[player.get_id()]))
        opponent_mobility = len(state._reachable_neighbours(state.rep.pawn_positions[opponent.get_id()]))

        return (
            (opponent_distance - own_distance)
            + 0.15 * wall_advantage
            + 0.05 * (own_mobility - opponent_mobility)
        )

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

        for wall in self._candidate_walls(state, deadline):
            self._check_time(deadline)
            if state._is_wall_legal(wall):
                yield StatelessAction({
                    "type": "horizontal" if wall.orientation == Orientation.HORIZONTAL else "vertical",
                    "destination": (wall.row, wall.col),
                })

    def _candidate_walls(self, state: GameStateQuoridor, deadline: float) -> set[Wall]:
        """Restreint les murs à ceux qui touchent le plus court chemin d'au
        moins un des deux joueurs.

        Un mur qui ne touche ni l'un ni l'autre chemin ne peut pas modifier
        la distance restante d'un joueur ; il est donc inutile de
        l'explorer. Cela réduit le facteur de branchement d'environ 150
        positions à quelques dizaines, ce qui permet d'augmenter la
        profondeur de recherche.
        """
        candidates: set[Wall] = set()
        for player in state.players:
            path = self._shortest_path_cells(state, player, deadline)
            if path is None:
                continue
            for start, end in zip(path, path[1:]):
                candidates.update(state._candidate_blocking_walls(start, end))
        return candidates

    def _shortest_path_cells(
        self, state: GameStateQuoridor, player: Player, deadline: float,
    ) -> list[tuple[int, int]] | None:
        """BFS retournant le plus court chemin (liste de cases) d'un joueur
        vers son but, en ignorant le pion adverse (comme state._shortest_path).
        """
        start = state.rep.pawn_positions[player.id]
        if start[0] == player.get_goal_row():
            return [start]

        queue = deque([start])
        visited = {start}
        parents: dict[tuple[int, int], tuple[int, int]] = {}

        while queue:
            self._check_time(deadline)
            position = queue.popleft()
            if position[0] == player.get_goal_row():
                path = [position]
                while position in parents:
                    position = parents[position]
                    path.append(position)
                path.reverse()
                return path

            for neighbour in state._reachable_neighbours(position):
                if neighbour not in visited:
                    visited.add(neighbour)
                    parents[neighbour] = position
                    queue.append(neighbour)

        return None

    @staticmethod
    def _check_time(deadline: float) -> None:
        if perf_counter() >= deadline:
            raise TimeoutError("Search time budget exhausted.")
