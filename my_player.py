# INF8175 - Projet Quoridor
# Marouane Istadad - 2346401
# Asmaa Sabouri - 2224168

import math
from collections import deque
from collections.abc import Iterator
from time import perf_counter

from actions_quoridor import Orientation, Wall
from board_quoridor import BoardQuoridor
from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from seahorse.player.player import Player


class MyPlayer(PlayerQuoridor):
    """Ouverture prudente, puis minimax alpha-bêta avec caches."""

    MAX_DEPTH = 4
    MAX_TIME_PER_MOVE = 5.0
    MAX_WALL_CANDIDATES = 12
    MAX_CACHE_ENTRIES = 20_000
    OPENING_LAST_STEP = 6
    OPENING_WALL_GAIN = 2

    def __init__(self, piece_type: str, goal_row: int = 0, name: str = "bob", *args, **kwargs) -> None:
        super().__init__(piece_type, goal_row, name, *args, **kwargs)
        self._transpositions = {}
        self._paths = {}

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15 * 60, **kwargs) -> Action:
        """Choisit une ouverture sûre ou le meilleur coup complètement recherché."""
        if remaining_time <= 0:
            raise RuntimeError("Time limit exceeded.")
        if current_state.is_done():
            raise RuntimeError("Game is already finished.")

        deadline = perf_counter() + min(self.MAX_TIME_PER_MOVE, remaining_time / 30)
        self._transpositions.clear()
        self._paths.clear()
        best_action = next(current_state.generate_possible_stateless_actions(), None)
        if best_action is None:
            raise RuntimeError("No legal action available.")

        try:
            opening_action = self._opening_action(current_state, deadline)
        except TimeoutError:
            return best_action
        if opening_action is not None:
            return opening_action

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

    def _opening_action(
        self, state: GameStateQuoridor, deadline: float,
    ) -> StatelessAction | None:
        """Avance sur plateau vide, sauf saut offert ou mur précoce rentable."""
        if state.step > self.OPENING_LAST_STEP or state.rep.walls:
            return None

        player = state.active_player
        if player.id != self.get_id():
            return None
        opponent = state._opponent(player)
        own_distance, _ = self._path_info(state, player, deadline)
        opponent_distance, _ = self._path_info(state, opponent, deadline)
        if own_distance is None or opponent_distance is None:
            return None
        # Si nous sommes déjà en retard, un mur peut valoir plus qu'un pas.
        if opponent_distance < own_distance - 1:
            return None

        row, col = state.rep.pawn_positions[player.id]
        goal_row = player.get_goal_row()
        if row == goal_row:
            return None
        forward = (row + (1 if goal_row > row else -1), col)
        action = next(
            (move for move in state._legal_moves() if move.data["destination"] == forward),
            None,
        )
        if action is None:
            return None

        next_state = state.apply_action(action)
        next_distance, _ = self._path_info(next_state, player, deadline)
        if next_distance != own_distance - 1:
            return None

        opponent_position = state.rep.pawn_positions[opponent.id]
        for reply in next_state._legal_moves():
            destination = reply.data["destination"]
            jump_length = sum(abs(destination[i] - opponent_position[i]) for i in (0, 1))
            if jump_length > 1:
                return None

        if state.rep.remaining_walls[player.id] > 0:
            for wall in self._candidate_walls(state, deadline):
                self._check_time(deadline)
                next_wall_state = self._legal_wall_state(state, wall, deadline)
                if next_wall_state is None:
                    continue
                own_after, _ = self._path_info(next_wall_state, player, deadline)
                opponent_after, _ = self._path_info(next_wall_state, opponent, deadline)
                gain = (opponent_after - opponent_distance) - (own_after - own_distance)
                if gain >= self.OPENING_WALL_GAIN:
                    return None

        return action

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
            return self.evaluate(current_state, deadline=deadline), None

        key = self._position_key(current_state)
        cached = self._transpositions.get(key)
        preferred = cached[3] if cached is not None else None
        if cached is not None and cached[0] >= depth:
            _, cached_value, bound, cached_action = cached
            if bound == "exact":
                return cached_value, cached_action
            if bound == "lower":
                alpha = max(alpha, cached_value)
            else:
                beta = min(beta, cached_value)
            if alpha >= beta:
                return cached_value, cached_action
        search_alpha, search_beta = alpha, beta

        best_value = -math.inf if is_maximizing else math.inf
        best_action = None

        for action in self._legal_actions(current_state, deadline, preferred):
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
            return self.evaluate(current_state, deadline=deadline), None

        bound = "exact"
        if best_value <= search_alpha:
            bound = "upper"
        elif best_value >= search_beta:
            bound = "lower"
        if cached is None or depth >= cached[0]:
            self._remember(self._transpositions, key, (depth, best_value, bound, best_action))
        return best_value, best_action

    @staticmethod
    def _position_key(state: GameStateQuoridor) -> tuple:
        board = state.rep
        return (
            board.dimension,
            tuple(sorted(board.pawn_positions.items())),
            board.walls,
            tuple(sorted(board.remaining_walls.items())),
            state.active_player.id,
        )

    def _remember(self, cache: dict, key: tuple, value: tuple) -> None:
        if key not in cache and len(cache) >= self.MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def evaluate(self, state: GameStateQuoridor, *, deadline: float = math.inf) -> float:
        """Favorise un chemin court vers notre but, un long chemin adverse,
        un avantage en murs restants et une bonne mobilité immédiate.
        """
        player = next(p for p in state.players if p.get_id() == self.get_id())
        opponent = next(p for p in state.players if p.get_id() != self.get_id())

        if state.scores[player.get_id()] == 1.0:
            return math.inf
        if state.scores[opponent.get_id()] == 1.0:
            return -math.inf

        own_distance, _ = self._path_info(state, player, deadline)
        opponent_distance, _ = self._path_info(state, opponent, deadline)
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

    def _legal_actions(
        self, state: GameStateQuoridor, deadline: float,
        preferred: StatelessAction | None = None,
    ) -> Iterator[StatelessAction]:
        """Joue le coup mémorisé, puis classe les déplacements et les murs."""
        if preferred is not None:
            self._check_time(deadline)
            yield preferred

        player = state.active_player
        opponent = state._opponent(player)
        own_distance, _ = self._path_info(state, player, deadline)
        moves = []
        for action in state._legal_moves():
            self._check_time(deadline)
            if preferred is not None and action.data == preferred.data:
                continue
            next_state = state.apply_action(action)
            if next_state.scores[player.id] == 1.0:
                priority = math.inf
            else:
                next_distance, _ = self._path_info(next_state, player, deadline)
                priority = own_distance - next_distance
            moves.append((priority, action))
        moves.sort(key=lambda item: -item[0])
        for _, action in moves:
            yield action

        if state.rep.remaining_walls[player.id] > 0:
            opponent_distance, _ = self._path_info(state, opponent, deadline)
            walls = []
            for wall in sorted(self._candidate_walls(state, deadline), key=lambda w: (w.row, w.col, w.orientation.value)):
                self._check_time(deadline)
                next_state = self._legal_wall_state(state, wall, deadline)
                if next_state is None:
                    continue
                next_own, _ = self._path_info(next_state, player, deadline)
                next_opponent, _ = self._path_info(next_state, opponent, deadline)
                # Un mur compte surtout s'il rallonge la route adverse sans
                # rallonger la nôtre. Le léger coût évite de gaspiller un mur.
                priority = (next_opponent - opponent_distance) - (next_own - own_distance) - 0.15
                walls.append((priority, wall))
            walls.sort(key=lambda item: -item[0])
            for priority, wall in walls[:self.MAX_WALL_CANDIDATES]:
                action = StatelessAction({
                    "type": "horizontal" if wall.orientation == Orientation.HORIZONTAL else "vertical",
                    "destination": (wall.row, wall.col),
                })
                if preferred is None or action.data != preferred.data:
                    yield action

    def _legal_wall_state(
        self, state: GameStateQuoridor, wall: Wall, deadline: float,
    ) -> GameStateQuoridor | None:
        """Reproduit la validation du moteur en réutilisant le cache BFS."""
        board = state.rep
        if board.remaining_walls[state.active_player.id] <= 0:
            return None
        if wall.orientation == Orientation.HORIZONTAL:
            if not (0 <= wall.col < board.dimension - 1):
                return None
            overlaps = (wall, Wall(wall.row, wall.col - 1, wall.orientation), Wall(wall.row, wall.col + 1, wall.orientation))
            crossing = Wall(wall.row - 1, wall.col + 1, Orientation.VERTICAL)
        else:
            if not (0 <= wall.row < board.dimension - 1):
                return None
            overlaps = (wall, Wall(wall.row - 1, wall.col, wall.orientation), Wall(wall.row + 1, wall.col, wall.orientation))
            crossing = Wall(wall.row + 1, wall.col - 1, Orientation.HORIZONTAL)
        if any(other in board.walls for other in overlaps) or crossing in board.walls:
            return None

        remaining = dict(board.remaining_walls)
        remaining[state.active_player.id] -= 1
        new_board = BoardQuoridor(board.pawn_positions, board.walls | {wall}, remaining, board.dimension)
        next_state = GameStateQuoridor(state.scores, state.active_player, state.players, new_board, state.step)
        for player in state.players:
            self._check_time(deadline)
            if self._path_info(next_state, player, deadline)[0] is None:
                return None
        return next_state

    def _candidate_walls(self, state: GameStateQuoridor, deadline: float) -> set[Wall]:
        """Couvre les chemins actuels et les sorties proches des deux pions."""
        candidates: set[Wall] = set()
        for player in state.players:
            path = self._shortest_path_cells(state, player, deadline)
            if path is None:
                continue
            for start, end in zip(path, path[1:]):
                candidates.update(state._candidate_blocking_walls(start, end))
            # Les murs autour des pions couvrent aussi des sorties proches
            # qui ne figurent pas dans le seul chemin choisi par le BFS.
            start = state.rep.pawn_positions[player.id]
            for end in state._reachable_neighbours(start):
                candidates.update(state._candidate_blocking_walls(start, end))
        return candidates

    def _shortest_path_cells(
        self, state: GameStateQuoridor, player: Player, deadline: float,
    ) -> list[tuple[int, int]] | None:
        """BFS retournant le plus court chemin (liste de cases) d'un joueur
        vers son but, en ignorant le pion adverse (comme state._shortest_path).
        """
        return self._path_info(state, player, deadline)[1]

    def _path_info(
        self, state: GameStateQuoridor, player: Player, deadline: float,
    ) -> tuple[int | None, tuple[tuple[int, int], ...] | None]:
        """Distance et chemin BFS mémorisés par murs, départ et objectif."""
        start = state.rep.pawn_positions[player.id]
        key = (state.rep.dimension, state.rep.walls, start, player.get_goal_row())
        cached = self._paths.get(key)
        if cached is not None:
            return cached
        if start[0] == player.get_goal_row():
            result = (0, (start,))
            self._remember(self._paths, key, result)
            return result

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
                result = (len(path) - 1, tuple(path))
                self._remember(self._paths, key, result)
                return result

            for neighbour in state._reachable_neighbours(position):
                if neighbour not in visited:
                    visited.add(neighbour)
                    parents[neighbour] = position
                    queue.append(neighbour)

        result = (None, None)
        self._remember(self._paths, key, result)
        return result

    @staticmethod
    def _check_time(deadline: float) -> None:
        if perf_counter() >= deadline:
            raise TimeoutError("Search time budget exhausted.")
