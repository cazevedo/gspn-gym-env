import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from gspn_lib import gspn_tools
import sys

class MultiGSPNenv_v1(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gspn_model=None, gspn_path=None, n_locations=None, n_robots=None, actions_maps=None,
                 reward_function=None, use_expected_time=False, verbose=False, idd=None):
        print('Multi GSPN Gym Env V1')
        self.id = idd
        self.verbose = verbose
        self.n_robots = n_robots
        self.n_locations = n_locations
        self.use_expected_time = use_expected_time
        self.actions_id_to_name = actions_maps[0]
        self.actions_name_to_id = actions_maps[1]

        if not reward_function:
            raise Exception('Please select one reward function: either 1 or 2')
        self.reward_function_type = reward_function

        if gspn_path != None:
            pn_tool = gspn_tools.GSPNtools()
            self.mr_gspn = pn_tool.import_greatspn(gspn_path)[0]
            # pn_tool.draw_gspn(mr_gspn)
        elif gspn_model != None:
            self.mr_gspn = gspn_model
        else:
            raise Exception('Please provide a GSPN object or a GSPN path of the environment model.')

        # Init timestamp
        self.timestamp = 0

        # [max_n_tokens_in_place0, max_n_tokens_in_place1, ... max_n_tokens_in_placen]
        # we approximate this to: [n_robots, n_robots, ... nrobots]
        self.observation_space = spaces.MultiDiscrete(nvec=[n_robots]*len(self.mr_gspn.get_current_marking()))
        n_actions = len(self.actions_id_to_name.keys())
        # {0,1,...,n_actions}
        self.action_space = spaces.Discrete(n_actions)

        self.enabled_parallel_transitions = {}

    def step(self, action):
        # get disabled actions in current state
        disabled_actions_names, disabled_actions_indexes = self.get_disabled_actions()

        # get current state
        current_state = self.get_current_state()
        if self.verbose:
            print('S: ', current_state)
            # print('Enabled Timed transitions : ', self.enabled_parallel_transitions)

        # map input action to associated transition
        if action in disabled_actions_indexes:
            transition = None
        else:
            transition = self.action_to_transition(action)
        if self.verbose:
            print('Action: ', action, transition)

        if transition != None:
            # apply action
            self.mr_gspn.fire_transition(transition)

            # get execution time (until the next decision state)
            # get also the sequence of the fired transitions ['t1', 't2', ...]
            elapsed_time, fired_transitions = self.execute_actions(use_expected_time=self.use_expected_time)

            reward = self.reward_function(current_state, transition, elapsed_time)

            # in a MRS the fired timed transition may not correspond to the selected action
            # this is the expected time that corresponds to the selected action
            action_expected_time = self.get_action_time(transition)

            self.timestamp += elapsed_time
        else:
            raise Exception('Disabled transition selected! This is not possible.')

        if self.verbose:
            print('Reward: ', reward)
            print('Timestamp: ', self.timestamp)
            print('Action expected time: ', action_expected_time)
            # print("S actions disabled: ", disabled_actions_names)

        # get enabled actions in the next state
        next_state_enabled_actions_names, next_state_enabled_actions_indexes = self.get_enabled_actions()

        # get next state
        next_state = self.marking_to_state()
        # next_state_string = self.get_current_state()
        if self.verbose:
            print("S': ", self.get_current_state())
            print("Available actions in s': ", next_state_enabled_actions_names)
            print()

        episode_done = False

        return next_state, reward, episode_done, \
               {'timestamp': self.timestamp,
                'disabled_actions': (disabled_actions_names, disabled_actions_indexes),
                'next_state_enabled_actions': (next_state_enabled_actions_names, next_state_enabled_actions_indexes),
                'action_time': action_expected_time,
                'fired_transitions': fired_transitions}

    def reset(self):
        self.timestamp = 0.0
        self.mr_gspn.reset_simulation()
        next_state = self.marking_to_state()
        self.enabled_parallel_transitions = {}

        # get enabled actions in the next state
        next_state_enabled_actions_names, next_state_enabled_actions_indexes = self.get_enabled_actions()

        return next_state, {'timestamp': self.timestamp, 'actions_info': [],
                               'disabled_actions': (None, None),
                               'next_state_enabled_actions': (
                               next_state_enabled_actions_names, next_state_enabled_actions_indexes),
                               'action_time': None}

    def render(self, mode='human'):
        raise Exception('Rendering not implemented')

    def close(self):
        self.reset()
        # print('Au Revoir Shoshanna!')

    def get_current_state(self):
        return self.mr_gspn.get_current_marking(sparse_marking=True)

    def action_to_transition(self, action):
        return self.actions_id_to_name[int(action)]

    def marking_to_state(self):
        # map dict marking to list marking
        marking_dict = self.mr_gspn.get_current_marking(sparse_marking=True)
        state = [0]*len(self.mr_gspn.get_current_marking().keys())
        for place_name, number_robots in marking_dict.items():
            token_index = self.mr_gspn.places_to_index[place_name]
            state[token_index] = number_robots

        return state

    def reward_function(self, sparse_state=None, transition=None, elapsed_time=0.0):
        if self.reward_function_type == 1:
            reward = 0.0

            if 'Insp' in transition:
                reward += 10.0

        elif self.reward_function_type == 2:
            reward = 0.0

            if 'Insp' in transition:
                reward += 500.0
            elif ('Charge' in transition) and (not ('Mobile' in transition)):
                reward += 100.0
        else:
            reward = 0.0

            if 'Insp' in transition:
                reward += 10.0

            robots_discharged = 0
            for local_state, robots in sparse_state.items():
                if 'Low' in local_state:
                    robots_discharged += robots

            reward -= robots_discharged*elapsed_time

        return reward

    def fire_timed_transitions(self, enabled_timed_transitions, use_expected_time):
        if use_expected_time:
            # convert the rate into expected time and store that transition if it was not already stored
            for tr_name, tr_rate in enabled_timed_transitions.copy().items():
                if tr_name not in self.enabled_parallel_transitions:
                    self.enabled_parallel_transitions[tr_name] = [1.0 / tr_rate]

                n_sampled_times = len(self.enabled_parallel_transitions[tr_name])
                tr_index = self.mr_gspn.transitions_to_index[tr_name]
                arcs_in = self.mr_gspn.get_arc_in_m()
                places_dict = self.mr_gspn.get_places()
                input_place_ratios = []
                sample_new_time = True
                for i, tr_coord in enumerate(arcs_in.coords[1]):
                    if tr_coord == tr_index:
                        place_index = arcs_in.coords[0][i]
                        place_name = self.mr_gspn.index_to_places[place_index]
                        n_tokens = places_dict[place_name]
                        arc_weight = arcs_in.data[i]
                        ratio = int(n_tokens/arc_weight)
                        # the ratio gives us the number of sampled times that must exist in the
                        # parallel dict, for this specific transition
                        input_place_ratios.append(ratio)
                        if ratio <= n_sampled_times:
                            sample_new_time = False
                            break
                # sample the amount necessary such that the number of
                # sampled times equals the smallest the place ratio
                if sample_new_time and len(input_place_ratios) > 0:
                    while len(self.enabled_parallel_transitions[tr_name]) < min(input_place_ratios):
                        self.enabled_parallel_transitions[tr_name].append(1.0 / tr_rate)

        else:
            # convert the rate into sampled elapsed time
            # sample from each exponential distribution prob_dist(x) = lambda * exp(-lambda * x)
            # in this case the beta rate parameter is used instead, where beta = 1/lambda
            # store enabled transition if it was not already stored
            for tr_name, tr_rate in enabled_timed_transitions.copy().items():
                if tr_name not in self.enabled_parallel_transitions:
                    self.enabled_parallel_transitions[tr_name] = [np.random.exponential(scale=(1.0 / tr_rate),
                                                                                        size=None)]

                n_sampled_times = len(self.enabled_parallel_transitions[tr_name])
                tr_index = self.mr_gspn.transitions_to_index[tr_name]
                arcs_in = self.mr_gspn.get_arc_in_m()
                places_dict = self.mr_gspn.get_places()
                input_place_ratios = []
                sample_new_time = True
                for i, tr_coord in enumerate(arcs_in.coords[1]):
                    if tr_coord == tr_index:
                        place_index = arcs_in.coords[0][i]
                        place_name = self.mr_gspn.index_to_places[place_index]
                        n_tokens = places_dict[place_name]
                        arc_weight = arcs_in.data[i]
                        ratio = int(n_tokens / arc_weight)
                        # the ratio gives us the number of sampled times that must exist in the
                        # parallel dict, for this specific transition
                        input_place_ratios.append(ratio)
                        if ratio <= n_sampled_times:
                            sample_new_time = False
                            break
                # sample the amount necessary such that the number of
                # sampled times equals the smallest place ratio
                if sample_new_time and len(input_place_ratios) > 0:
                    while len(self.enabled_parallel_transitions[tr_name]) < min(input_place_ratios):
                        self.enabled_parallel_transitions[tr_name].append(np.random.exponential(scale=(1.0 / tr_rate),
                                                                                                size=None))
        # delete the transitions that were enabled, didn't fire and are not longer enabled
        disabled_transitions = set(self.enabled_parallel_transitions.keys())-set(enabled_timed_transitions.keys())
        for tr_name in disabled_transitions:
            del self.enabled_parallel_transitions[tr_name]

        # select the transition with the lowest execution time
        execution_time = np.inf
        for tr_name, tr_time in self.enabled_parallel_transitions.items():
            new_min_time = min(tr_time)
            if new_min_time < execution_time:
                timed_transition = tr_name
                execution_time = new_min_time

        transitions_to_fire = []
        transitions_to_fire.append(timed_transition)

        # delete transition to be fired
        if len(self.enabled_parallel_transitions[timed_transition]) > 1:
            self.enabled_parallel_transitions[timed_transition].remove(execution_time)
        else:
            del self.enabled_parallel_transitions[timed_transition]

        # decreased elapsed time for the remaining enabled transitions
        for tr_name, tr_exp_time in self.enabled_parallel_transitions.copy().items():
            new_tr_time = list(np.array(tr_exp_time) - execution_time)
            if any(i <= 0 for i in new_tr_time):
                # according to PN formalism there are never two timed transitions
                # with the same elapsed time
                # instead we should sum a very small time (e.g. 1e-6)
                # to ensure that only 1 transition fires at each time
                # when using expected time this arises more often
                pruned_new_tr_time = []
                for remaining_time in new_tr_time:
                    if remaining_time <= 0:
                        # transitions_to_fire.append(tr_name)
                        pruned_new_tr_time.append(1e-6)
                    else:
                        pruned_new_tr_time.append(remaining_time)

                if len(pruned_new_tr_time) > 0:
                    self.enabled_parallel_transitions[tr_name] = pruned_new_tr_time
                else:
                    del self.enabled_parallel_transitions[tr_name]
            else:
                self.enabled_parallel_transitions[tr_name] = new_tr_time

        for transition_name in transitions_to_fire:
            self.mr_gspn.fire_transition(transition_name)
        return execution_time, transitions_to_fire

    def fire_random_switch(self, random_switch):
        if len(random_switch) > 1:
            s = sum(random_switch.values())
            random_switch_id = list(random_switch.keys())
            random_switch_prob = np.zeros(len(random_switch))
            # normalize the associated probabilities
            for idx, tr_info in enumerate(random_switch.items()):
                tr_name = tr_info[0]
                tr_weight = tr_info[1]
                random_switch_id[idx] = tr_name
                random_switch_prob[idx] = tr_weight / s

            # Draw from all enabled immediate transitions
            firing_transition = np.random.choice(a=random_switch_id, size=None, p=random_switch_prob)
        else:
            # Fire the only available immediate transition
            firing_transition = list(random_switch.keys())[0]

        self.mr_gspn.fire_transition(firing_transition)

    def check_actions_state(self, enabled_imm_transitions):
        action_enabled = False
        random_switch_available = False
        for tr_name, tr_rate in enabled_imm_transitions.items():
            if tr_rate == 0:
                action_enabled = True
            elif tr_rate != 0:
                random_switch_available = True
        return action_enabled, random_switch_available

    def execute_actions(self, use_expected_time=False):
        enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        # check if there is at least one imm transition with weight != 0 and check if there is one with weight == 0
        enabled_actions, random_switch = self.check_actions_state(enabled_imm_transitions)

        elapsed_time = 0
        fired_transitions = []
        while random_switch or (not enabled_actions):
            while random_switch:
                self.fire_random_switch(enabled_imm_transitions)
                enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()
                enabled_actions, random_switch = self.check_actions_state(enabled_imm_transitions)

            while (enabled_timed_transitions and not enabled_actions and not random_switch):
                action_elapsed_time, tr_fired = self.fire_timed_transitions(enabled_timed_transitions,
                                                                            use_expected_time)
                elapsed_time += action_elapsed_time
                fired_transitions += tr_fired
                enabled_timed_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()
                enabled_actions, random_switch = self.check_actions_state(enabled_imm_transitions)

        return elapsed_time, fired_transitions

    def get_disabled_actions(self):
        enabled_actions_names, enabled_actions_indexes = self.get_enabled_actions()

        disabled_actions_indexes = list(set(self.actions_id_to_name.keys()) - set(enabled_actions_indexes))
        disabled_actions_names = list(set(self.actions_name_to_id.keys()) - set(enabled_actions_names))

        return disabled_actions_names, disabled_actions_indexes

    def get_enabled_actions(self):
        enabled_exp_transitions, enabled_imm_transitions = self.mr_gspn.get_enabled_transitions()

        enabled_actions_indexes = []
        enabled_actions_names = []
        for tr_name, tr_rate in enabled_imm_transitions.items():
            if tr_rate == 0:
                enabled_actions_names.append(tr_name)
                enabled_actions_indexes.append(self.actions_name_to_id[tr_name])

        return enabled_actions_names, enabled_actions_indexes

    def get_action_time(self, fired_transition):
        associated_timed_tr = fired_transition + '_Finished'
        transition_rate = self.mr_gspn.get_transition_rate(associated_timed_tr)
        action_expected_time = 1.0/transition_rate
        return action_expected_time

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]