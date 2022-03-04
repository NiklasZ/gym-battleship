import gym
import numpy as np
from gym import spaces
from copy import deepcopy
from typing import Union
from typing import Tuple
from typing import Optional
from collections import namedtuple

Ship = namedtuple('Ship', ['min_x', 'max_x', 'min_y', 'max_y'])
Action = namedtuple('Action', ['x', 'y'])

# TODO these should be configurable (also change observation space)
EMPTY, HIT, MISS, SUNK = 1, 2, 3, 4


def is_notebook():
    """Helper used to change the way the environment in rendered"""
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        else:
            return False  # Terminal running IPython or other types
    except NameError:
        return False  # Probably standard Python interpreter


class BattleshipEnv(gym.Env):
    """
    # TODO more detailed description
    0: unexplored, meaning no missile has been fired here yet.
    1: a missile has been fired here and hit nothing
    2: a missile has been fired here and hit a ship
    3: a missile has been fired here and the affected ship has sunk.
    """

    def __init__(self,
                 board_size: Tuple = None,
                 ship_sizes: dict = None,
                 episode_steps: int = 100,
                 reward_dictionary: Optional[dict] = None):

        self.ship_sizes = ship_sizes or {5: 1, 4: 1, 3: 2, 2: 1}
        self.board_size = board_size or (10, 10)

        # (m x n) matrix of board_size where each cell is either 0 (empty) or 1 (contains ship).
        # This state is hidden from the player.
        self.board = None
        # (m x n) matrix of board_size, where each cell has a state EMPTY,HIT,MISS,SUNK. This state is visible to the player.
        self.observation = None
        self.board_generated = None  # Hidden state generated and left not updated (for debugging purposes)
        self.ship_positions = None  # Ship[] list containing list of ship positions
        self.remaining_ships = None  # Dict of ships that haven't been sunk yet.

        self.done = None
        self.step_count = None
        self.episode_steps = episode_steps
        self.batched = False  # Unrelated flag that some machine learning frameworks check for.

        reward_dictionary = {} if reward_dictionary is None else reward_dictionary
        default_reward_dictionary = {  # todo further tuning of the rewards required
            'win': 100,  # for sinking all ships
            'missed': 0,  # for missing a shot
            'hit': 1,  # for hitting a ship
            'repeat_missed': -1,  # for shooting at an already missed cell
            'repeat_hit': -0.5  # for shooting at an already hit cell
        }
        # use default entries + whatever is provided as input
        self.reward_dictionary = default_reward_dictionary | reward_dictionary

        self.action_space = spaces.Discrete(self.board_size[0] * self.board_size[1])
        self.observation_space = spaces.Box(low=EMPTY, high=SUNK,
                                            shape=(self.board_size[0], self.board_size[1]), dtype=int)

    def step(self, input_action: Union[int, tuple, np.ndarray]) -> Tuple[np.ndarray, int, bool, dict]:

        if isinstance(input_action, np.ndarray):
            size = input_action.size
            assert 1 <= size <= 2, f'action numpy array must be size 1 or 2. Received {size}'
            raw_action = np.asscalar(input_action) if size == 1 else (input_action[0], input_action[1])
        else:
            raw_action = input_action

        if isinstance(raw_action, (int, np.integer)):
            limit = self.board_size[0] * self.board_size[1]
            assert (0 <= raw_action < limit), \
                f"Invalid action (The encoded action {raw_action} is outside of the board limits {limit})"
            action = Action(x=raw_action % self.board_size[0], y=raw_action // self.board_size[0])

        elif isinstance(raw_action, tuple):
            assert (0 <= raw_action[0] < self.board_size[0] and 0 <= raw_action[1] < self.board_size[1]), \
                f"Invalid action (The action {raw_action} " \
                f"is outside the board limits ({self.board_size[0]},{self.board_size[1]}))"
            action = Action(x=raw_action[0], y=raw_action[1])

        else:
            raise AssertionError(
                f"Invalid action (Unsupported raw_action type: {type(raw_action)}, value: {raw_action})")

        self.step_count += 1

        # Check if the game is done (if true, the current step is the "last step")
        if self.step_count >= self.episode_steps:
            self.done = True

        # Hit (board[x, y] == 1)
        if self.board[action.x, action.y] == 1:
            self.board[action.x, action.y] = 0

            self.observation[action.x, action.y] = HIT

            ship = self._find_hit_ship(action)
            # If all parts of the ship were hit, it sinks
            if np.all(self.observation[ship.min_x:ship.max_x, ship.min_y:ship.max_y] == HIT):
                self.observation[ship.min_x:ship.max_x, ship.min_y:ship.max_y] = SUNK
                ship_size = max(ship.max_x - ship.min_x, ship.max_y - ship.min_y)
                self.remaining_ships[ship_size] -= 1

            # Win (No boat left)
            if not self.board.any():
                self.done = True
                return self.observation, self.reward_dictionary['win'], self.done, self.remaining_ships
            return self.observation, self.reward_dictionary['hit'], self.done, self.remaining_ships

        # Repeat hit or sink (observation[x, y] == HIT,SUNK)
        elif self.observation[action.x, action.y] in [HIT, SUNK]:
            return self.observation, self.reward_dictionary['repeat_hit'], self.done, self.remaining_ships

        # Repeat missed (observation[x, y] == MISS)
        elif self.observation[action.x, action.y] == MISS:
            return self.observation, self.reward_dictionary['repeat_missed'], self.done, self.remaining_ships

        # Missed (Action not repeated and boat(s) not hit)
        else:
            self.observation[action.x, action.y] = MISS
            return self.observation, self.reward_dictionary['missed'], self.done, self.remaining_ships

    def reset(self) -> np.ndarray:
        self._set_board()
        self.board_generated = deepcopy(self.board)
        self.observation = np.full(self.board_size, EMPTY, dtype=np.int32)
        self.remaining_ships = deepcopy(self.ship_sizes)
        self.step_count = 0
        self.done = False
        return self.observation

    def _set_board(self) -> None:
        self.board = np.zeros(self.board_size, dtype=np.int32)
        self.ship_positions = []
        for ship_size, ship_count in self.ship_sizes.items():
            for _ in range(ship_count):
                self.ship_positions.append(self._place_ship(ship_size))

    def _place_ship(self, ship_size: int) -> Ship:
        can_place_ship = False
        ship = None
        while not can_place_ship:  # todo add protection infinite loop
            ship = self._get_ship(ship_size, self.board_size)
            can_place_ship = self._is_place_empty(ship)
        self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y] = True
        return ship

    @staticmethod
    def _get_ship(ship_size: int, board_size: tuple) -> Ship:
        if np.random.choice(('Horizontal', 'Vertical')) == 'Horizontal':
            min_x = np.random.randint(0, board_size[0] + 1 - ship_size)
            min_y = np.random.randint(0, board_size[1])
            return Ship(min_x=min_x, max_x=min_x + ship_size, min_y=min_y, max_y=min_y + 1)
        else:
            min_x = np.random.randint(0, board_size[0])
            min_y = np.random.randint(0, board_size[1] + 1 - ship_size)
            return Ship(min_x=min_x, max_x=min_x + 1, min_y=min_y, max_y=min_y + ship_size)

    def _is_place_empty(self, ship: Ship) -> bool:
        return np.count_nonzero(self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y]) == 0

    def _find_hit_ship(self, action: Action) -> Union[None, Ship]:

        def ship_filter(s: Ship):
            return s.min_x <= action.x < s.max_x and s.min_y <= action.y < s.max_y

        matches = list(filter(ship_filter, self.ship_positions))
        # Sanity-check. Can probably delete later.
        assert (len(matches) == 1), \
            f"Error: Expected to find exactly 1 ship on cell {action}. Found: {matches}"
        return matches[0]

    # TODO re-render example images for README.md
    def render(self, mode='human'):
        board = np.empty(self.board_size, dtype=str)
        board[self.observation == HIT] = '🞇'
        board[self.observation == SUNK] = '❌'
        board[self.observation == MISS] = '⚫'
        self._render(board)

    def render_board_generated(self):
        board = np.empty(self.board_size, dtype=str)
        board[self.board_generated != 0] = '⬛'
        self._render(board)

    # TODO in-place render could look nice (create GIF from it).
    @staticmethod
    def _render(board, symbol='⬜'):
        import pandas as pd

        num_rows, num_columns = board.shape
        columns = [chr(i) for i in range(ord('A'), ord('A') + num_columns)]
        index = [i + 1 for i in range(num_rows)]

        dataframe = pd.DataFrame(board, columns=columns, index=index)
        dataframe = dataframe.replace([''], symbol)

        if is_notebook():
            from IPython.display import display
            display(dataframe)
        else:
            print(dataframe, end='\n')

        # todo maybe put the board generated on the right side
        #  https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
