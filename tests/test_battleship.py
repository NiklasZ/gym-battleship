import gym
import gym_battleship
import pytest
import numpy as np

from gym_battleship import BattleshipEnv
from gym_battleship.environments.battleship import Ship

e = BattleshipEnv()
UNKNOWN = e.observation_dictionary['unknown']
HIT = e.observation_dictionary['hit']
MISS = e.observation_dictionary['missed']
SUNK = e.observation_dictionary['sunk']


def test_initialisation_1():
    """
    Initialises correctly with default parameters
    """
    env = BattleshipEnv()
    assert env.ship_sizes == {5: 1, 4: 1, 3: 2, 2: 1}
    assert env.board_size == (10, 10)
    assert env.reward_dictionary == {
        'win': 100,
        'missed': 0,
        'hit': 1,
        'repeat_missed': -1,
        'repeat_hit': -0.5
    }
    assert env.episode_steps == 100

    env.reset()
    assert np.all(env.observation == UNKNOWN)
    assert env.remaining_ships == env.ship_sizes
    assert np.count_nonzero(env.board_generated) == 5 + 4 + 3 * 2 + 2


def test_initialisation_2():
    """
    Initialises correctly with custom parameters.
    """
    env = BattleshipEnv(board_size=(9, 9),
                        ship_sizes={3: 1},
                        episode_steps=20,
                        reward_dictionary={
                            'win': 10,
                            'missed': -0.25,
                        },
                        observation_dictionary={'unknown': 0.5})
    assert env.ship_sizes == {3: 1}
    assert env.board_size == (9, 9)
    assert env.reward_dictionary == {
        'win': 10,
        'missed': -0.25,
        'hit': 1,
        'repeat_missed': -1,
        'repeat_hit': -0.5

    }
    assert env.episode_steps == 20

    env.reset()
    assert np.all(env.observation == 0.5)
    assert env.remaining_ships == env.ship_sizes
    assert np.count_nonzero(env.board_generated) == 3


def test_miss_ship():
    env = BattleshipEnv(board_size=(5, 5),
                        ship_sizes={3: 1},
                        reward_dictionary={
                            'missed': -0.25,
                            'repeat_missed': -0.5,
                        })
    env.reset()
    # Will manually set the ship position as randomised tests are awful to design around.
    env.board = np.array([[0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    env.ship_positions = [Ship(min_x=0, max_x=2, min_y=4, max_y=5)]

    # 1st shot
    state, reward, done, remaining_ships = env.step((0, 0))
    assert np.count_nonzero(state == MISS) == 1
    assert state[0, 0] == MISS
    assert reward == -0.25
    assert not done
    assert remaining_ships == {3: 1}

    # Aim at same spot again
    state, reward, done, remaining_ships = env.step((0, 0))
    assert np.count_nonzero(state == MISS) == 1
    assert state[0, 0] == MISS
    assert reward == -0.5
    assert not done
    assert remaining_ships == {3: 1}


def test_hit_ship():
    env = BattleshipEnv(board_size=(5, 5),
                        ship_sizes={3: 1},
                        reward_dictionary={
                            'hit': 1,
                            'repeat_hit': -0.5,
                        })
    env.reset()
    # Will manually set the ship position as randomised tests are awful to design around.
    env.board = np.array([[0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    env.ship_positions = [Ship(min_x=0, max_x=2, min_y=4, max_y=5)]

    # 1st shot
    state, reward, done, remaining_ships = env.step((0, 4))
    assert np.count_nonzero(state == HIT) == 1
    assert state[0, 4] == 2
    assert reward == 1
    assert not done
    assert remaining_ships == {3: 1}

    # Aim at same spot again
    state, reward, done, remaining_ships = env.step((0, 4))
    assert np.count_nonzero(state == HIT) == 1
    assert state[0, 4] == 2
    assert reward == -0.5
    assert not done
    assert remaining_ships == {3: 1}


def test_sink_ship():
    env = BattleshipEnv(board_size=(5, 5),
                        ship_sizes={2: 2},
                        reward_dictionary={
                            'hit': 1,
                            'repeat_hit': -0.5,
                        })
    env.reset()
    # Will manually set the ship position as randomised tests are awful to design around.
    env.board = np.array([[1, 1, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    env.ship_positions = [Ship(min_x=0, max_x=1, min_y=0, max_y=2), Ship(min_x=0, max_x=2, min_y=4, max_y=5)]

    # 1st shot
    state, reward, done, remaining_ships = env.step((0, 0))
    assert np.count_nonzero(state == HIT) == 1
    assert state[0, 0] == HIT
    assert reward == 1
    assert not done
    assert remaining_ships == {2: 2}

    # 2nd shot
    state, reward, done, remaining_ships = env.step((0, 1))
    assert np.count_nonzero(state == SUNK) == 2
    assert np.count_nonzero(state == HIT) == 0
    assert state[0, 0] == SUNK and state[0, 1] == SUNK
    assert reward == 1
    assert not done
    assert remaining_ships == {2: 1}

    # Aim at same spot again
    state, reward, done, remaining_ships = env.step((0, 1))
    assert np.count_nonzero(state == SUNK) == 2
    assert state[0, 1] == SUNK
    assert reward == -0.5
    assert not done
    assert remaining_ships == {2: 1}


def test_game_over_turn_limit():
    env = BattleshipEnv(episode_steps=2)
    env.reset()
    state, reward, done, remaining_ships = env.step((0, 0))
    assert not done
    state, reward, done, remaining_ships = env.step((0, 0))
    assert done


def test_game_over_by_victory():
    env = BattleshipEnv(board_size=(5, 5),
                        ship_sizes={2: 1})
    env.reset()
    # Will manually set the ship position as randomised tests are awful to design around.
    env.board = np.array([[1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    env.ship_positions = [Ship(min_x=0, max_x=1, min_y=0, max_y=2)]

    # 1st shot
    state, reward, done, remaining_ships = env.step((0, 0))
    assert np.count_nonzero(state == HIT) == 1
    assert state[0, 0] == HIT
    assert reward == 1
    assert not done
    assert remaining_ships == {2: 1}

    # 2nd shot
    state, reward, done, remaining_ships = env.step((0, 1))
    assert np.count_nonzero(state == SUNK) == 2
    assert state[0, 0] == SUNK and state[0, 1] == SUNK
    assert reward == 100
    assert done
    assert remaining_ships == {2: 0}
