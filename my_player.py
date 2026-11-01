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
    """Ouverture prudente, puis minimax alpha-bêta tactique avec caches."""

    MAX_DEPTH = 4
    MAX_TIME_PER_MOVE = 7.0
    MAX_WALL_CANDIDATES = 6
    MAX_TACTICAL_ACTIONS = 3
    MAX_TACTICAL_EXTENSIONS = 1
    MAX_CACHE_ENTRIES = 20_000
    OPENING_LAST_STEP = 6
    OPENING_WALL_GAIN = 2
    TACTICAL_WALL_GAIN = 2
    NEAR_GOAL_DISTANCE = 3
    WALL_BASE_VALUE = 0.35
    LAST_WALL_RESERVE_VALUE = 3.0
    WALL_THREAT_WEIGHT = 0.45
    PREVENTION_WEIGHT = 0.8
    RACE_WIN_BONUS = 6.0
    DISTANCE_SOFT_CAP = 8

    def __init__(self, piece_type: str, goal_row: int = 0, name: str = "bob", *args, **kwargs) -> None:
        super().__init__(piece_type, goal_row, name, *args, **kwargs)
        self._transpositions = {}
        self._paths = {}
        self._pawn_paths = {}
        self._path_edges = {}
        self._ordered_actions = {}
        self._action_metadata = {}
        self._active_wall_threats = {}
        self._prevention_bonuses = {}

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15 * 60, **kwargs) -> Action:
        """Choisit une ouverture sûre ou le meilleur coup complètement recherché."""
        if remaining_time <= 0:
            raise RuntimeError("Time limit exceeded.")
        if current_state.is_done():
            raise RuntimeError("Game is already finished.")

        deadline = perf_counter() + min(self.MAX_TIME_PER_MOVE, remaining_time / 30)
        self._transpositions.clear()
        self._paths.clear()
        self._pawn_paths.clear()
        self._path_edges.clear()
        self._ordered_actions.clear()
        self._action_metadata.clear()
        self._active_wall_threats.clear()
        self._prevention_bonuses.clear()
        try:
            best_action = self._fallback_action(current_state, deadline)
        except TimeoutError:
            best_action = next(
                current_state.generate_possible_stateless_actions(), None,
            )
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
                    extensions_left=self.MAX_TACTICAL_EXTENSIONS,
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
        extensions_left: int = 0,
    ) -> tuple[float, Action | None]:
        """Renvoie (score, action), toujours du point de vue de ce joueur.

        MAX choisit notre meilleur coup ; MIN choisit la réponse adverse
        la plus défavorable. Aux feuilles, l'action vaut None.
        """
        self._check_time(deadline)
        if current_state.is_done():
            return self.evaluate(current_state, deadline=deadline), None

        is_extension = depth <= 0
        if is_extension:
            if extensions_left <= 0:
                return self.evaluate(current_state, deadline=deadline), None
            actions = self._tactical_actions(current_state, deadline)
            if not actions:
                return self.evaluate(current_state, deadline=deadline), None
        else:
            actions = None

        position_key = self._position_key(current_state)
        key = (position_key, extensions_left)
        cached = None if is_extension else self._transpositions.get(key)
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

        action_iterator = (
            actions if is_extension
            else self._legal_actions(current_state, deadline, preferred)
        )
        for action in action_iterator:
            new_state = current_state.apply_action(action)
            value, _ = self.minimax(
                new_state, max(0, depth - 1), alpha, beta, not is_maximizing,
                deadline=deadline,
                extensions_left=(extensions_left - 1 if is_extension else extensions_left),
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
        if not is_extension and (cached is None or depth >= cached[0]):
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

    def _remember(self, cache: dict, key: tuple, value: object) -> None:
        if key not in cache and len(cache) >= self.MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def evaluate(self, state: GameStateQuoridor, *, deadline: float = math.inf) -> float:
        """Évalue la course, le tempo, les réserves et les menaces de murs."""
        player = next(p for p in state.players if p.get_id() == self.get_id())
        opponent = next(p for p in state.players if p.get_id() != self.get_id())

        if state.scores[player.get_id()] == 1.0:
            return math.inf
        if state.scores[opponent.get_id()] == 1.0:
            return -math.inf

        own_static, _ = self._path_info(state, player, deadline)
        opponent_static, _ = self._path_info(state, opponent, deadline)
        if own_static is None:
            return -math.inf
        if opponent_static is None:
            return math.inf

        # Ces distances tiennent compte des sauts et diagonales autour du pion
        # adverse. Elles donnent une meilleure estimation des courses serrées.
        own_race = self._pawn_distance(state, player, deadline)
        opponent_race = self._pawn_distance(state, opponent, deadline)
        if own_race is None:
            own_race = own_static
        if opponent_race is None:
            opponent_race = opponent_static

        raw_margin = opponent_race - own_race
        path_score = self._soft_distance_margin(raw_margin)

        own_walls = state.rep.remaining_walls[player.id]
        opponent_walls = state.rep.remaining_walls[opponent.id]
        wall_score = (
            self._wall_inventory_value(
                own_walls, opponent_race,
            )
            - self._wall_inventory_value(
                opponent_walls, own_race,
            )
        )

        own_mobility = len(
            self._pawn_neighbours(
                state, player, state.rep.pawn_positions[player.id],
            )
        )
        opponent_mobility = len(
            self._pawn_neighbours(
                state, opponent, state.rep.pawn_positions[opponent.id],
            )
        )

        # La menace mémorisée est le meilleur mur immédiatement disponible
        # pour le joueur au trait. Elle pénalise les positions où une clôture
        # peut ajouter plusieurs cases au prochain coup.
        threat = self._active_wall_threats.get(self._position_key(state), 0.0)
        threat_score = self.WALL_THREAT_WEIGHT * threat
        if state.active_player.id != player.id:
            threat_score = -threat_score

        # Sans murs, la parité du trait décide souvent une course pourtant
        # évaluée à égalité par les seules distances.
        race_score = 0.0
        if own_walls == 0 and opponent_walls == 0:
            own_plies = 2 * own_race - (
                1 if state.active_player.id == player.id else 0
            )
            opponent_plies = 2 * opponent_race - (
                1 if state.active_player.id == opponent.id else 0
            )
            if own_plies < opponent_plies:
                race_score = self.RACE_WIN_BONUS
            elif own_plies > opponent_plies:
                race_score = -self.RACE_WIN_BONUS
        else:
            race_score = 0.2 if state.active_player.id == player.id else -0.2

        return (
            path_score
            + wall_score
            + threat_score
            + race_score
            + 0.08 * (own_mobility - opponent_mobility)
        )

    def _legal_actions(
        self, state: GameStateQuoridor, deadline: float,
        preferred: StatelessAction | None = None,
    ) -> Iterator[StatelessAction]:
        """Classe globalement déplacements, murs tactiques et préventifs."""
        key = self._position_key(state)
        actions = self._ordered_actions.get(key)
        if actions is None:
            actions = self._build_ordered_actions(state, deadline)
            self._remember(self._ordered_actions, key, actions)

        if preferred is not None:
            self._check_time(deadline)
            yield preferred

        for action in actions:
            self._check_time(deadline)
            if preferred is None or action.data != preferred.data:
                yield action

    def _build_ordered_actions(
        self, state: GameStateQuoridor, deadline: float,
    ) -> tuple[StatelessAction, ...]:
        """Construit une liste courte, tactique et bien ordonnée."""
        player = state.active_player
        opponent = state._opponent(player)
        own_distance, _ = self._path_info(state, player, deadline)
        opponent_distance, _ = self._path_info(state, opponent, deadline)
        own_race = self._pawn_distance(state, player, deadline)

        ranked: list[tuple[float, StatelessAction]] = []
        metadata: dict[tuple, tuple[float, float, float, bool]] = {}

        for action in state._legal_moves():
            self._check_time(deadline)
            next_state = state.apply_action(action)
            terminal = next_state.scores[player.id] == 1.0
            if terminal:
                priority = math.inf
            else:
                next_distance, _ = self._path_info(next_state, player, deadline)
                next_race = self._pawn_distance(next_state, player, deadline)
                if own_race is None:
                    own_race = own_distance
                if next_race is None:
                    next_race = next_distance
                priority = (own_distance - next_distance) + 0.5 * (own_race - next_race)
            ranked.append((priority, action))
            metadata[self._action_key(action)] = (priority, 0.0, 0.0, terminal)

        best_threat = 0.0
        if state.rep.remaining_walls[player.id] > 0:
            prevention = self._preventive_wall_bonuses(state, player, deadline)
            candidates = self._candidate_walls(state, deadline) | set(prevention)
            wall_cost = self._wall_spend_cost(
                state, player, own_distance, opponent_distance,
            )
            walls: list[tuple[float, float, float, Wall]] = []
            for wall in sorted(
                candidates,
                key=lambda item: (item.row, item.col, item.orientation.value),
            ):
                self._check_time(deadline)
                next_state = self._legal_wall_state(state, wall, deadline)
                if next_state is None:
                    continue
                next_own, _ = self._path_info(next_state, player, deadline)
                next_opponent, _ = self._path_info(next_state, opponent, deadline)
                own_increase = next_own - own_distance
                opponent_increase = next_opponent - opponent_distance
                net_gain = opponent_increase - own_increase
                preventive_gain = prevention.get(wall, 0.0)
                priority = (
                    net_gain
                    + self.PREVENTION_WEIGHT * preventive_gain
                    - wall_cost
                )
                walls.append((priority, opponent_increase, preventive_gain, wall))
                best_threat = max(best_threat, net_gain, 0.5 * opponent_increase)

            walls.sort(
                key=lambda item: (
                    -item[0], -item[1], item[3].row, item[3].col,
                    item[3].orientation.value,
                )
            )
            selected_walls = walls[:self.MAX_WALL_CANDIDATES]
            for priority, opponent_increase, preventive_gain, wall in selected_walls:
                action = StatelessAction({
                    "type": "horizontal" if wall.orientation == Orientation.HORIZONTAL else "vertical",
                    "destination": (wall.row, wall.col),
                })
                ranked.append((priority, action))
                metadata[self._action_key(action)] = (
                    priority, opponent_increase, preventive_gain, False,
                )

        ranked.sort(
            key=lambda item: (
                -item[0],
                self._action_key(item[1]),
            )
        )
        key = self._position_key(state)
        self._remember(self._action_metadata, key, metadata)
        self._remember(self._active_wall_threats, key, best_threat)
        return tuple(action for _, action in ranked)

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
        """Couvre tous les plus courts chemins, les sorties et les clôtures."""
        candidates: set[Wall] = set()
        for player in state.players:
            for start, end in self._shortest_path_edges(state, player, deadline):
                candidates.update(state._candidate_blocking_walls(start, end))
            start = state.rep.pawn_positions[player.id]
            for end in state._reachable_neighbours(start):
                candidates.update(state._candidate_blocking_walls(start, end))

        # Une clôture devient souvent dangereuse à son dernier segment. On
        # considère donc ses extensions ainsi que le mur perpendiculaire qui
        # empêcherait chacune d'elles d'être posée.
        for extension in self._wall_extensions(state):
            candidates.add(extension)
            candidates.add(self._crossing_wall(extension))
        return candidates

    def _preventive_wall_bonuses(
        self, state: GameStateQuoridor, defender: Player, deadline: float,
    ) -> dict[Wall, float]:
        """Valorise les croisements qui neutralisent une fermeture adverse."""
        key = (self._position_key(state), defender.id)
        cached = self._prevention_bonuses.get(key)
        if cached is not None:
            return cached

        attacker = state._opponent(defender)
        if (
            state.rep.remaining_walls[defender.id] <= 0
            or state.rep.remaining_walls[attacker.id] <= 0
        ):
            self._remember(self._prevention_bonuses, key, {})
            return {}

        attacker_state = GameStateQuoridor(
            state.scores, attacker, state.players, state.rep, state.step,
        )
        defender_before, _ = self._path_info(state, defender, deadline)
        attacker_before, _ = self._path_info(state, attacker, deadline)
        bonuses: dict[Wall, float] = {}

        for extension in self._wall_extensions(state):
            self._check_time(deadline)
            next_state = self._legal_wall_state(attacker_state, extension, deadline)
            if next_state is None:
                continue
            defender_after, _ = self._path_info(next_state, defender, deadline)
            attacker_after, _ = self._path_info(next_state, attacker, deadline)
            damage = defender_after - defender_before
            attacker_cost = max(0, attacker_after - attacker_before)
            net_damage = damage - attacker_cost
            if damage >= self.TACTICAL_WALL_GAIN and net_damage > 0:
                crossing = self._crossing_wall(extension)
                bonuses[crossing] = max(
                    bonuses.get(crossing, 0.0),
                    net_damage,
                    0.5 * damage,
                )

        self._remember(self._prevention_bonuses, key, bonuses)
        return bonuses

    @staticmethod
    def _wall_extensions(state: GameStateQuoridor) -> set[Wall]:
        """Renvoie les segments susceptibles de prolonger les murs présents."""
        extensions: set[Wall] = set()
        dimension = state.rep.dimension
        for wall in state.rep.walls:
            if wall.orientation == Orientation.HORIZONTAL:
                for col in (wall.col - 2, wall.col + 2):
                    if 0 <= wall.row < dimension and 0 <= col < dimension - 1:
                        extensions.add(Wall(wall.row, col, wall.orientation))
            else:
                for row in (wall.row - 2, wall.row + 2):
                    if 0 <= row < dimension - 1 and 0 <= wall.col < dimension:
                        extensions.add(Wall(row, wall.col, wall.orientation))
        return extensions

    @staticmethod
    def _crossing_wall(wall: Wall) -> Wall:
        """Renvoie le mur perpendiculaire qui croise exactement ``wall``."""
        if wall.orientation == Orientation.HORIZONTAL:
            return Wall(wall.row - 1, wall.col + 1, Orientation.VERTICAL)
        return Wall(wall.row + 1, wall.col - 1, Orientation.HORIZONTAL)

    def _shortest_path_edges(
        self, state: GameStateQuoridor, player: Player, deadline: float,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
        """Renvoie les arêtes appartenant à au moins un plus court chemin."""
        start = state.rep.pawn_positions[player.id]
        goal_row = player.get_goal_row()
        key = (state.rep.dimension, state.rep.walls, start, goal_row)
        if key in self._path_edges:
            return self._path_edges[key]

        def distances(sources: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
            queue = deque(sources)
            result = {source: 0 for source in sources}
            while queue:
                self._check_time(deadline)
                position = queue.popleft()
                for neighbour in state._reachable_neighbours(position):
                    if neighbour not in result:
                        result[neighbour] = result[position] + 1
                        queue.append(neighbour)
            return result

        from_start = distances([start])
        goal_cells = [(goal_row, col) for col in range(state.rep.dimension)]
        to_goal = distances(goal_cells)
        shortest = to_goal.get(start)
        edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        if shortest is not None:
            for position, prefix in from_start.items():
                for neighbour in state._reachable_neighbours(position):
                    suffix = to_goal.get(neighbour)
                    if suffix is not None and prefix + 1 + suffix == shortest:
                        edges.add((position, neighbour))

        result = tuple(sorted(edges))
        self._remember(self._path_edges, key, result)
        return result

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

    def _pawn_distance(
        self, state: GameStateQuoridor, player: Player, deadline: float,
    ) -> int | None:
        """Distance BFS incluant les sauts et diagonales autour de l'adversaire."""
        start = state.rep.pawn_positions[player.id]
        opponent = state._opponent(player)
        opponent_position = state.rep.pawn_positions[opponent.id]
        key = (
            state.rep.dimension, state.rep.walls, start,
            opponent_position, player.get_goal_row(),
        )
        if key in self._pawn_paths:
            return self._pawn_paths[key]

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            self._check_time(deadline)
            position, distance = queue.popleft()
            if position[0] == player.get_goal_row():
                self._remember(self._pawn_paths, key, distance)
                return distance
            for neighbour in self._pawn_neighbours(state, player, position):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, distance + 1))

        self._remember(self._pawn_paths, key, None)
        return None

    def _pawn_neighbours(
        self, state: GameStateQuoridor, player: Player,
        position: tuple[int, int],
    ) -> tuple[tuple[int, int], ...]:
        """Mouvements légaux d'un pion, quel que soit le joueur actif."""
        opponent = state._opponent(player)
        opponent_position = state.rep.pawn_positions[opponent.id]
        dimension = state.rep.dimension
        row, col = position
        result: list[tuple[int, int]] = []

        for row_delta, col_delta in state.DIRECTIONS:
            adjacent = (row + row_delta, col + col_delta)
            if not (
                0 <= adjacent[0] < dimension
                and 0 <= adjacent[1] < dimension
            ):
                continue
            if state._blocked_by_wall(position, adjacent):
                continue
            if adjacent != opponent_position:
                result.append(adjacent)
                continue

            behind = (
                adjacent[0] + row_delta,
                adjacent[1] + col_delta,
            )
            if (
                0 <= behind[0] < dimension
                and 0 <= behind[1] < dimension
                and not state._blocked_by_wall(adjacent, behind)
            ):
                result.append(behind)
                continue

            perpendicular = (
                ((-1, 0), (1, 0))
                if row_delta == 0
                else ((0, -1), (0, 1))
            )
            for side_row, side_col in perpendicular:
                diagonal = (adjacent[0] + side_row, adjacent[1] + side_col)
                if (
                    0 <= diagonal[0] < dimension
                    and 0 <= diagonal[1] < dimension
                    and not state._blocked_by_wall(adjacent, diagonal)
                ):
                    result.append(diagonal)

        return tuple(dict.fromkeys(result))

    def _tactical_actions(
        self, state: GameStateQuoridor, deadline: float,
    ) -> tuple[StatelessAction, ...]:
        """Prolonge seulement gains, menaces et défenses de clôture."""
        actions = tuple(self._legal_actions(state, deadline))
        metadata = self._action_metadata.get(self._position_key(state), {})
        player = state.active_player
        opponent = state._opponent(player)
        own_distance = self._pawn_distance(state, player, deadline)
        opponent_distance = self._pawn_distance(state, opponent, deadline)
        near_goal = (
            own_distance is not None
            and own_distance <= self.NEAR_GOAL_DISTANCE
        )
        opponent_near = (
            opponent_distance is not None
            and opponent_distance <= self.NEAR_GOAL_DISTANCE
        )

        selected: list[StatelessAction] = []
        best_moves: list[StatelessAction] = []
        tactical_position = near_goal or opponent_near
        for action in actions:
            priority, wall_gain, prevention, terminal = metadata.get(
                self._action_key(action), (0.0, 0.0, 0.0, False),
            )
            if terminal:
                selected.append(action)
                tactical_position = True
            elif action.data["type"] == "move":
                if len(best_moves) < 2:
                    best_moves.append(action)
            elif (
                wall_gain >= self.TACTICAL_WALL_GAIN
                or prevention >= self.TACTICAL_WALL_GAIN
                or (opponent_near and wall_gain >= 1)
            ):
                selected.append(action)
                tactical_position = True

        if not tactical_position:
            return ()

        for action in best_moves:
            if all(action.data != current.data for current in selected):
                selected.append(action)
        order = {self._action_key(action): index for index, action in enumerate(actions)}
        selected.sort(key=lambda action: order[self._action_key(action)])
        return tuple(selected[:self.MAX_TACTICAL_ACTIONS])

    def _fallback_action(
        self, state: GameStateQuoridor, deadline: float,
    ) -> StatelessAction | None:
        """Repli sûr : victoire immédiate, sinon meilleur progrès de pion."""
        moves = state._legal_moves()
        if not moves:
            return next(state.generate_possible_stateless_actions(), None)
        player = state.active_player
        best_action = moves[0]
        best_distance = math.inf
        for action in moves:
            self._check_time(deadline)
            next_state = state.apply_action(action)
            if next_state.scores[player.id] == 1.0:
                return action
            distance = self._pawn_distance(next_state, player, deadline)
            if distance is not None and distance < best_distance:
                best_distance = distance
                best_action = action
        return best_action

    def _wall_spend_cost(
        self, state: GameStateQuoridor, player: Player,
        own_distance: int, opponent_distance: int,
    ) -> float:
        """Coût marginal croissant pour les derniers murs et l'overblocking."""
        opponent = state._opponent(player)
        remaining = state.rep.remaining_walls[player.id]
        opponent_walls = state.rep.remaining_walls[opponent.id]
        cost = self.WALL_BASE_VALUE
        if opponent_walls > 0:
            if remaining == 1:
                cost += self.LAST_WALL_RESERVE_VALUE
            elif remaining == 2:
                cost += 1.0
        lead = opponent_distance - own_distance
        if lead >= self.DISTANCE_SOFT_CAP:
            cost += 0.3 * (lead - self.DISTANCE_SOFT_CAP + 1)
        return cost

    def _wall_inventory_value(
        self, walls: int, target_distance: int,
    ) -> float:
        """Valeur dynamique d'un stock, avec prime d'accès au dernier mur."""
        if walls <= 0:
            return 0.0
        proximity = max(0, 6 - target_distance)
        value = walls * (self.WALL_BASE_VALUE + 0.08 * proximity)
        # Le premier mur disponible garde une valeur tactique propre : il
        # peut encore créer un détour ou interdire une fermeture, même si
        # l'autre joueur vient d'épuiser son stock.
        value += self.LAST_WALL_RESERVE_VALUE
        return value

    def _soft_distance_margin(self, margin: int) -> float:
        """Réduit le rendement des cases ajoutées à une avance déjà énorme."""
        magnitude = abs(margin)
        if magnitude <= self.DISTANCE_SOFT_CAP:
            return float(margin)
        softened = self.DISTANCE_SOFT_CAP + 0.35 * (
            magnitude - self.DISTANCE_SOFT_CAP
        )
        return softened if margin > 0 else -softened

    @staticmethod
    def _action_key(action: StatelessAction) -> tuple[str, tuple[int, int]]:
        return action.data["type"], tuple(action.data["destination"])

    @staticmethod
    def _check_time(deadline: float) -> None:
        if perf_counter() >= deadline:
            raise TimeoutError("Search time budget exhausted.")
