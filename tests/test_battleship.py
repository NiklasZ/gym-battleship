import gym
import gym_battleship
import pytest
import numpy as np

from gym_battleship import BattleshipEnv
from gym_battleship.environments.battleship import Ship

e = BattleshipEnv()


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
    assert np.all(env.observation == 0)
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
                        get_remaining_ships=True,
                        get_invalid_action_mask=True)
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
    state, reward, done = env.step((0, 0))
    assert np.count_nonzero(state[..., 0]) == 1
    assert state[0, 0, 0] == 1
    assert np.count_nonzero(state[..., 1]) == 1
    assert state[0, 0, 1] == 1
    assert reward == -0.25
    assert not done

    # Aim at same spot again
    state, reward, done = env.step((0, 0))
    assert np.count_nonzero(state[..., 0]) == 1
    assert state[0, 0, 0] == 1
    assert np.count_nonzero(state[..., 1]) == 1
    assert state[0, 0, 1] == 1
    assert reward == -0.5
    assert not done


def test_miss_ship_2():
    # Same as before but we ask for extra details in the observation.
    env = BattleshipEnv(board_size=(5, 5),
                        ship_sizes={3: 1},
                        reward_dictionary={
                            'missed': -0.25,
                            'repeat_missed': -0.5,
                        },
                        get_remaining_ships=True,
                        get_invalid_action_mask=True)
    env.reset()
    # Will manually set the ship position as randomised tests are awful to design around.
    env.board = np.array([[0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    env.ship_positions = [Ship(min_x=0, max_x=2, min_y=4, max_y=5)]

    # 1st shot
    observation, reward, done = env.step((0, 0))
    state, remaining_ships, valid_actions = observation['observation'], observation['remaining_ships'], observation[
        'valid_actions']
    assert np.count_nonzero(state[..., 0]) == 1
    assert state[0, 0, 0] == 1
    assert np.count_nonzero(state[..., 1]) == 1
    assert state[0, 0, 1] == 1
    assert reward == -0.25
    assert not done
    assert remaining_ships == {3: 1}
    assert np.all(valid_actions == np.array([[False, True, True, True, True],
                                            [True, True, True, True, True],
                                            [True, True, True, True, True],
                                            [True, True, True, True, True],
                                            [True, True, True, True, True]]))

    # Aim at same spot again
    observation, reward, done = env.step((0, 0))
    state, remaining_ships, valid_actions = observation['observation'], observation['remaining_ships'], observation[
        'valid_actions']
    assert np.count_nonzero(state[..., 0]) == 1
    assert state[0, 0, 0] == 1
    assert np.count_nonzero(state[..., 1]) == 1
    assert state[0, 0, 1] == 1
    assert reward == -0.5
    assert not done
    assert remaining_ships == {3: 1}
    assert np.all(valid_actions == np.array([[False, True, True, True, True],
                                            [True, True, True, True, True],
                                            [True, True, True, True, True],
                                            [True, True, True, True, True],
                                            [True, True, True, True, True]]))

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
    state, reward, done = env.step((0, 4))
    assert np.count_nonzero(state[..., 0] == 1) == 1
    assert state[0, 4, 0] == 1
    assert np.count_nonzero(state[..., 2] == 1) == 1
    assert state[0, 4, 2] == 1
    assert reward == 1
    assert not done

    # Aim at same spot again
    state, reward, done = env.step((0, 4))
    assert np.count_nonzero(state[..., 0] == 1) == 1
    assert state[0, 4, 0] == 1
    assert np.count_nonzero(state[..., 2] == 1) == 1
    assert state[0, 4, 2] == 1
    assert reward == -0.5
    assert not done


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
    state, reward, done = env.step((0, 0))
    assert np.count_nonzero(state[..., 0] == 1) == 1
    assert state[0, 0, 0] == 1
    assert np.count_nonzero(state[..., 2] == 1) == 1
    assert state[0, 0, 2] == 1
    assert reward == 1
    assert not done

    # 2nd shot
    state, reward, done = env.step((0, 1))
    assert np.count_nonzero(state[..., 0] == 1) == 2
    assert np.count_nonzero(state[..., 2] == 1) == 0
    assert np.count_nonzero(state[..., 3] == 1) == 2
    assert state[0, 0, 3] == 1 and state[0, 1, 3] == 1
    assert reward == 1
    assert not done

    # Aim at same spot again
    state, reward, done = env.step((0, 1))
    assert np.count_nonzero(state[..., 0] == 1) == 2
    assert reward == -0.5
    assert not done


def test_game_over_turn_limit():
    env = BattleshipEnv(episode_steps=2)
    env.reset()
    state, reward, done = env.step((0, 0))
    assert not done
    state, reward, done = env.step((0, 0))
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
    state, reward, done = env.step((0, 0))
    assert np.count_nonzero(state[..., 0] == 1) == 1
    assert reward == 1
    assert not done

    # 2nd shot
    state, reward, done = env.step((0, 1))
    assert np.count_nonzero(state[..., 0] == 1) == 2
    assert np.count_nonzero(state[..., 3] == 1) == 2
    assert reward == 100
    assert done
