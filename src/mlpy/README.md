# mlpy
a python machine learning package


within each subfolder (also a python submodule) are different types of ml algorithms. So far here are the following submodules:

1) activation_functions: functions that are useful as activation functions for ml algorithms

2) bayesian: so far this submodule contains a naive bayes classifier.
             This submodule also has tests and examples, in the tests/ and examples/ subfolders.
             There is a test dataset (for spam filtering) in the examples/ directory, to run the
             spam filtering example, type 'python nb_spam_filtering.py' into the terminal.
             There are other files that can be run:
                 a) examples/nb_weather_play_outside.py if you want to see if you should go outside
                                                        and play given the weather.
                 b) test/test_nb.py if you want to run the naive bayes unit tests.
                 c) test/test_nbnode.py if you want to run the naive bayes node unit tests.

              Alternatively, you could type 'nosetests' in the terminal after navigating to
              the test/ directory.

3) data_partitioning: so far this module has some helpful functions for partitioning your datasets
                      between different targets.

4) nets: so far this submodule contains a artificial neural network.
         This submodule also has a test directory, although this does not have unit tests.
         There are several python files that solve toy problems, but be warned you will need
         matplotlib to run these files.

5) processes: so far this submodule only has the class template for a hidden markov model...stay tuned.

6) qlearn: so far this submodule has an untested q-learning algorithm using the artificial neural
           network from the nets submodule. It is my goal to use this algorithm to play an Atari
           game.

Depedencies:
    1) numpy REQUIRED FOR ALL
    2) random REQUIRED FOR ALL
    3) unittest *for only the bayesian test files
    4) matplotlib *for only test files in the bayesian, and nets submodules
    5) re *for only the bayesian example files
    

