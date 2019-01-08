# SYSTEM IMPORTS
import numpy
import random


# PYTHON PROJECT IMPORTS


class QLearner(object):
    def __init__(self, world_handle, seed=None):
        random.seed(seed=seed)
        self._world_handle = world_handle

    def learn_online(self, epochs, iterations_per_epoch, q_function, update_function,
                     discount_factor=1.0, random_action=0.01, log=True):
        for epoch in range(0, epochs):
            if log:
                print("Training epoch: %s" % epoch)
            self._world_handle.reset_world()
            collective_reward = 0.0

            iteration = 0
            while iteration < iterations_per_epoch and not self._world_handle.is_done():
                action_to_apply = None
                max_q_value = float("-inf")

                # select an action by choosing action that maximizes q function
                # (with a 'random_action' chance of choosing a random action
                if random.uniform(0.0, 1.0) < random_action:
                    action_to_apply = int(random.uniform(0,
                        len(self._world_handle.get_available_actions() - 1))
                    max_q_value = q_function(self._world_handle.get_current_state(), action_to_apply)
                else:
                    current_q_value = 0.0
                    for action in range(len(self._world_handle.get_available_actions())):
                        current_q_value = q_function(self._world_handle.get_current_state(), action)
                        if current_q_value > max_q_value:
                            max_q_value = current_q_value
                            action_to_apply = action

                 # apply the action
                 current_reward = self._world_handle.apply_action(action_to_apply)

                 # compute the 'target Q value'
                 # (current_reward + discount_factor * (max_a' q_function(new_state, a'))
                 error = current_reward + discount_factor *\
                     max([q_function (self._world_handle.get_current_state(), action)
                          for action in range(len(self._world_handle.get_available_actions() - 1))])

                 update_function(error)

    def learn_offline(self, epochs, iterations_per_epoch, q_function, update_function,
                      discount_factor=1.0, random_action=0.01, log=True):

        for epoch in range(0, epochs):
            training_features = None
            training_annotations = None
            if log:
                print("Training epoch: %s" % epoch)
            self._world_handle.reset_world()
            collective_reward = 0.0

            iteration = 0
            while iteration < iterations_per_second and not self._world_handle.is_done():
                action_to_apply = None
                max_q_value = float("-inf")
                current_state = self._world_handle.get_current_state()

                # select an action by choosing action that maximizes q function
                # (with a 'random_action' chance of choosing a random action
                if random.uniform(0.0, 1.0) < random_action:
                    action_to_apply = int(random.uniform(0,
                        len(self._world_handle.get_available_actions() - 1))
                    max_q_value = q_function(current_state, action_to_apply)
                else:
                    current_q_value = 0.0
                    for action in range(len(self._world_handle.get_available_actions())):
                        current_q_value = q_function(self._world_handle.get_current_state(), action)
                        if current_q_value > max_q_value:
                            max_q_value = current_q_value
                            action_to_apply = action

                # apply the action
                current_reward = self._world_handle.apply_action(action_to_apply)

                # compute the 'target Q value'
                # (current_reward + discount_factor * (max_a' q_function(new_state, a'))
                error = current_reward + discount_factor *\
                    max([q_function (self._world_handle.get_current_state(), action)
                         for action in range(len(self._world_handle.get_available_actions() - 1))])

                if training_features is None:
                    training_features = numpy.concatenate((current_state,
                        numpy.array([[action_to_apply]], dtype=float)), axis=1)
                else:
                    training_features = numpy.concatenate((training_features, 
                        numpy.concatenate((current_state, numpy.array([[action_to_apply]], dtype=float)),
                            axis=1)), axis=0)

                if training_annotations is None:
                    training_annotations = numpy.array([[error]])
                else:
                    training_annotations = numpy.concatenate((training_annotations,
                                                              numpy.array([[error]], dtype=float)), axis=0)

                update_function(training_features, training_annotations)

