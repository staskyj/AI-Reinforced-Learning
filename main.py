import gym
from gym import spaces
import pygame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class GridWorldEnv(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

  def __init__(self, render_mode=None, size=8):
    self.size = size  # The size of the square grid
    self.window_size = 512  # The size of the PyGame window

    # Observations are dictionaries with the agent's and the target's location.
    # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
    #self.observation_space = spaces.Discrete(5)

    self.observation_space = spaces.Dict({
      "agent1":
      spaces.Box(0, size - 1, shape=(3, ), dtype=int),
      "agent2":
      spaces.Box(0, size - 1, shape=(3, ), dtype=int),
      "target1":
      spaces.Box(0, size - 1, shape=(3, ), dtype=int),
      "target2":
      spaces.Box(0, size - 1, shape=(3, ), dtype=int),
      "PUblock1":
      spaces.Box(0, size - 1, shape=(3, ), dtype=int),
      "PUblock2":
      spaces.Box(0, size - 1, shape=(3, ), dtype=int),
      "risk_block1":
      spaces.Box(0, size - 1, shape=(3, ), dtype=int),
      "risk_block2":
      spaces.Box(0, size - 1, shape=(3, ), dtype=int),
    })
    self.observation_space = spaces.Discrete(8)

    # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
    self.action_space = spaces.Discrete(8)
    """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
    self._action_to_direction = {
      0: np.array([1, 0, 0]),
      1: np.array([0, 1, 0]),
      2: np.array([-1, 0, 0]),
      3: np.array([0, -1, 0]),
      4: np.array([0, 0, 1]),
      5: np.array([0, 0, -1]),
      6: np.array([0, 0, 0]),
      7: np.array([0, 0, 0])
      #6 and 7 will act as a pick up and dropoff action Always take this if possible
    }

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
    self.window = None
    self.clock = None
    self.risk_block1 = self.np_random.integers(0, self.size, size=3, dtype=int)
    self.risk_block2 = self.np_random.integers(0, self.size, size=3, dtype=int)
    self.PUblock1 = self.np_random.integers(0, self.size, size=3, dtype=int)
    self.PUblock2 = self.np_random.integers(0, self.size, size=3, dtype=int)

  def _get_obs(self):
    #return {"agent1": self._agent_location1, "target1": self._target_location1}
    return {
      "agent1": self._agent_location1,
      "agent2": self._agent_location2,
      "target1": self._target_location1,
      "target2": self._target_location2
    }

  def _get_info(self):
    return {
      "distance1":
      np.linalg.norm(self._agent_location1 - self._target_location1,
                     ord=1,
                     axis=0),
      "distance2":
      np.linalg.norm(self._agent_location2 - self._target_location2,
                     ord=1,
                     axis=0)
    }

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    # Choose the agent's location uniformly at random
    self._agent_location1 = self.np_random.integers(0,
                                                    self.size,
                                                    size=3,
                                                    dtype=int)
    self._agent_location2 = self._agent_location1
    while np.array_equal(self._agent_location2, self._agent_location1):
      self._agent_location2 = self.np_random.integers(0,
                                                      self.size,
                                                      size=3,
                                                      dtype=int)

#     self.risk_block1 = self.np_random.integers(0, self.size, size=3, dtype=int)
#     self.risk_block2 = self.np_random.integers(0, self.size, size=3, dtype=int)
#     self.PUblock1 = self.np_random.integers(0, self.size, size=3, dtype=int)
#     self.PUblock2 = self.np_random.integers(0, self.size, size=3, dtype=int)
# We will sample the target's location randomly until it does not coincide with the agent's location
    self._target_location1 = self._agent_location1
    while np.array_equal(self._target_location1, self._agent_location1):
      self._target_location1 = self.np_random.integers(0,
                                                       self.size,
                                                       size=3,
                                                       dtype=int)
    self._target_location2 = self._agent_location2
    while np.array_equal(self._target_location2, self._agent_location2):
      self._target_location2 = self.np_random.integers(0,
                                                       self.size,
                                                       size=3,
                                                       dtype=int)

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, info

  def step(self, action):
    # Map the action (element of {0,1,2,3}) to the direction we walk in
    reward1 = 0
    reward2 = 0
    if action not in self._action_to_direction:
      action = self.action_space.sample()
    #print(f"ACTION: {action}")

    direction = self._action_to_direction[action]
    PU_agent1 = False
    PU_agent2 = False
    done_agent1 = False
    done_agent2 = False
    terminated = False

    new_location1 = np.clip(self._agent_location1 + direction, 0,
                            self.size - 1)
    new_location2 = np.clip(self._agent_location2 + direction, 0,
                            self.size - 1)
    # # We use `np.clip` to make sure we don't leave the grid

    if (np.array_equal(new_location2, self._agent_location1)):
      reward1 -= 1
    else:
      self._agent_location2 = new_location2

    if (np.array_equal(new_location1, self._agent_location2)):
      reward1 -= 1
    else:
      self._agent_location1 = new_location1

    if (np.array_equal(self._agent_location1, self.risk_block1)
        or np.array_equal(self._agent_location1, self.risk_block2)):
      reward1 += -2
    if (np.array_equal(self._agent_location2, self.risk_block1)
        or np.array_equal(self._agent_location2, self.risk_block2)):
      reward2 += -2
    # An episode is done iff the agent has reached the target
    #terminated = (np.array_equal(self._agent_location1, self._target_location1) and np.array_equal(self._agent_location2, self._target_location2)) # Binary sparse rewards
    if (np.array_equal(self._agent_location1, self.PUblock1)):
      reward1 += 1
      PU_agent1 = True
    if (np.array_equal(self._agent_location2, self.PUblock2)):
      reward2 += 1
      PU_agent2 = True
    if (PU_agent1 == True
        and np.array_equal(self._agent_location1, self._target_location1)):
      done_agent1 = True
      reward1 += 2
    if (PU_agent2 == True
        and np.array_equal(self._agent_location2, self._target_location2)):
      done_agent2 = True
      reward2 += 2
    if (done_agent1 == True and done_agent2 == True):
      terminated = True
    else:
      reward1 = reward1 + 0
      reward2 = reward2 + 0

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, reward1, reward2, terminated, info

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()

  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
        (self.window_size, self.window_size))
    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255, 255, 255))
    pix_square_size = (self.window_size / self.size
                       )  # The size of a single grid square in pixels

    # First we draw the target
    pygame.draw.rect(
      canvas,
      (255, 0, 0),
      pygame.Rect(
        pix_square_size * self._target_location1,
        (pix_square_size, pix_square_size),
      ),
      pygame.Rect(
        pix_square_size * self._target_location2,
        (pix_square_size, pix_square_size),
      ),
    )
    # Now we draw the agent
    pygame.draw.circle(
      canvas,
      (0, 0, 255),
      (self._agent_location1 + 0.5) * pix_square_size,
      pix_square_size / 3,
    )

    # Finally, add some gridlines
    for x in range(self.size + 1):
      pygame.draw.line(
        canvas,
        0,
        (0, pix_square_size * x),
        (self.window_size, pix_square_size * x),
        width=3,
      )
      pygame.draw.line(
        canvas,
        0,
        (pix_square_size * x, 0),
        (pix_square_size * x, self.window_size),
        width=3,
      )

    if self.render_mode == "human":
      # The following line copies our drawings from `canvas` to the visible window
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      # We need to ensure that human-rendering occurs at the predefined framerate.
      # The following line will automatically add a delay to keep the framerate stable.
      self.clock.tick(self.metadata["render_fps"])
    else:  # rgb_array
      return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),
                          axes=(1, 0, 2))

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()


#random state function
def random(env,
           qtable,
           episodes,
           alpha,
           gamma,
           max_steps,
           state_agent1=None,
           state_agent2=None):
  for episodes in range(1, episodes + 1):
    state, info = env.reset()
    done = False
    score = 0
    steps = 0
    state_agent1 = list(state.values())[0]
    state_agent2 = list(state.values())[1]
    while not done and steps <= max_steps:
      action_agent1 = env.action_space.sample()
      action_agent2 = env.action_space.sample()
      new_state1, reward1, reward2, done1, info1 = env.step(action_agent1)
      new_state2, reward1, reward2, done2, info2 = env.step(action_agent2)
      new_state_agent1 = list(new_state1.values())[0]
      new_state_agent2 = list(new_state2.values())[1]
      new_action_1 = env.action_space.sample()
      new_action_2 = env.action_space.sample()
      #score2 += reward2
      # print(f"State agent 1: {state_agent1}")
      # print(f"action 1: {action_agent1}")

      # print(f"State agent 2: {state_agent2}")
      # print(f"action 2: {action_agent2}")
      # print(f"Step: {steps}")

      #this is the start of regular qlearning input table
      qtable[state_agent1, action_agent1] = qtable[state_agent1, action_agent1] + \
                               alpha * (reward1 + gamma * np.max(qtable[new_state_agent1, :]) - qtable[state_agent1, action_agent1])

      qtable[state_agent2, action_agent2] = qtable[state_agent2, action_agent2] + \
                               alpha * (reward2 + gamma * np.max(qtable[new_state_agent2, :]) - qtable[state_agent2, action_agent2])
      #this is the end of regular q learning input table

      #       #This is the start of SARSA algo implementation
      #       qtable[state_agent1,
      #              action_agent1] = qtable[state_agent1, action_agent1] + alpha * (
      #                reward1 + gamma * qtable[new_state_agent1, new_action_1] -
      #                qtable[state_agent1, action_agent1])
      #       qtable[state_agent2,
      #              action_agent2] = qtable[state_agent2, action_agent2] + alpha * (
      #                reward2 + gamma * qtable[new_state_agent2, new_action_2] -
      #                qtable[state_agent2, action_agent2])
      #       #This is the end of SARSA implemintation

      state_agent1 = new_state_agent1
      state_agent2 = new_state_agent2
      score += reward1 + reward2
      steps += 1
      if done1:
        break
  print()
  print('===========================================')
  print('Q-table after training with PRANDOM:')
  print(qtable)
  return qtable, state_agent1, state_agent2


#exploit function
def exploit(env, qtable, episodes, alpha, gamma, max_steps, state_agent1,
            state_agent2):

  steps = 0
  for episodes in range(1, episodes + 1):
    state, info = env.reset()
    print(state)
    print(info)
    done = False
    score = 0
    state_agent1 = list(state.values())[0]
    state_agent2 = list(state.values())[1]

    while not done and steps <= max_steps:
      if np.random.uniform(0, 100) < 80:
        if np.max(qtable[state_agent1]) > 0:
          action_agent1 = np.argmax(qtable[state_agent1, :])
          if (action_agent1 > 7):
            action_agent1 = env.action_space.sample()
        else:
          action_agent1 = env.action_space.sample()

        if np.max(qtable[state_agent2]) > 0:
          action_agent2 = np.argmax(qtable[state_agent2, :])
          if (action_agent2 > 7):
            action_agent2 = env.action_space.sample()
        else:
          action_agent2 = env.action_space.sample()
      else:
        action_agent1 = env.action_space.sample()
        action_agent2 = env.action_space.sample()

      new_state1, reward1, reward2, done1, info1 = env.step(action_agent1)
      new_state2, reward1, reward2, done2, info2 = env.step(action_agent2)
      new_state_agent1 = list(new_state1.values())[0]
      new_state_agent2 = list(new_state2.values())[1]
      new_action_1 = env.action_space.sample()
      new_action_2 = env.action_space.sample()

      # print(f"State agent 1: {state_agent1}")
      # print(f"action 1: {action_agent1}")

      # print(f"State agent 2: {state_agent2}")
      # print(f"action 2: {action_agent2}")
      # print(f"Step: {steps}")

      #this is the start of regular qlearning input table

      qtable[state_agent1, action_agent1] = qtable[state_agent1, action_agent1] + \
                                alpha * (reward1 + gamma * np.max(qtable[new_state_agent1, :]) - qtable[state_agent1, action_agent1])

      qtable[state_agent2, action_agent2] = qtable[state_agent2, action_agent2] + \
                                alpha * (reward2 + gamma * np.max(qtable[new_state_agent2, :]) - qtable[state_agent2, action_agent2])
      #this is the end of regular q learning input table

      #       #This is the start of SARSA algo implementation
      #       qtable[state_agent1,
      #              action_agent1] = qtable[state_agent1, action_agent1] + alpha * (
      #                reward1 + gamma * qtable[new_state_agent1, new_action_1] -
      #                qtable[state_agent1, action_agent1])
      #       qtable[state_agent2,
      #              action_agent2] = qtable[state_agent2, action_agent2] + alpha * (
      #                reward2 + gamma * qtable[new_state_agent2, new_action_2] -
      #                qtable[state_agent2, action_agent2])
      #       #This is the end of SARSA implemintation

      state_agent1 = new_state_agent1
      state_agent2 = new_state_agent2
      score += reward1 + reward2
      steps += 1
      if done:
        break

  print()
  print('===========================================')
  print('Q-table after training with PEXPLOIT:')
  print(qtable)
  return qtable, state_agent1, state_agent2


def main():
  env = GridWorldEnv()
  nb_states = env.observation_space.n
  nb_actions = env.action_space.n
  qtable = np.zeros((nb_states, nb_actions))
  q_out = np.zeros((nb_states, nb_actions))
  qtable1 = np.zeros((nb_states, nb_actions))
  qtable2 = np.zeros((nb_states, nb_actions))
  qtable3 = np.zeros((nb_states, nb_actions))
  qtable4 = np.zeros((nb_states, nb_actions))
  qtable5 = np.zeros((nb_states, nb_actions))
  qtable6 = np.zeros((nb_states, nb_actions))

  # Hyperparameters
  episodes = 1  # Total number of episodes
  alpha = 0.3  # Learning rate
  gamma = 0.5  # Discount factor
  max_steps1 = 500
  max_steps2 = 9500

  #exp_prob = 1           #exploreation probability

  print('Q-table before training:')
  print(qtable)
  #choose to run form here

  #First run with PRANDOM for 500 steps
  qtable1, state_agent1, state_agent2 = random(env, qtable, episodes, alpha,
                                               gamma, max_steps1)
  print("1st run")
  print(env.PUblock1)

  print(env.PUblock2)
  #2nd run with PEXPLOIT for 9500 steps
  qtable2, state_agent1, state_agent2 = exploit(env, qtable1, episodes, alpha,
                                                gamma, max_steps2,
                                                state_agent1, state_agent2)
  print("2nd run")
  print(env.PUblock1)

  print(env.PUblock2)
  #3rd run with PEXPLOIT for 9500 steps
  qtable3, state_agent1, state_agent2 = exploit(env, qtable2, episodes, alpha,
                                                gamma, max_steps2,
                                                state_agent1, state_agent2)
  print("3rd run")

  #After the 3rd run, change pickup locations to (2,3,3) and (1,3,1)
  env.PUblock1 = (2, 3, 3)
  env.PUblock2 = (1, 3, 1)

  print(env.PUblock1)

  print(env.PUblock2)
  #4rd run with PEXPLOIT using new pickup locations and 9500 steps
  qtable4, state_agent1, state_agent2 = exploit(env, qtable3, episodes, alpha,
                                                gamma, max_steps2,
                                                state_agent1, state_agent2)
  print("4th run")

  #5th run with PEXPLOIT using new pickup locations and 9500 steps
  qtable5, state_agent1, state_agent2 = exploit(env, qtable4, episodes, alpha,
                                                gamma, max_steps2,
                                                state_agent1, state_agent2)
  print("5th run")

  #6th run with PEXPLOIT using new pickup locations and 9500 steps
  qtable6, state_agent1, state_agent2 = exploit(env, qtable5, episodes, alpha,
                                                gamma, max_steps2,
                                                state_agent1, state_agent2)
  print("6th run")

  return qtable1, qtable2, qtable3, qtable4, qtable5, qtable6
  
if __name__ == "__main__":
  qtable1, qtable2, qtable3, qtable4, qtable5, qtable6 = main()