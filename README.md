# Smooth Policy Iteration (SPI)
The repository is developed for smooth policy iteration and its application in vehicle tracking tasks.
The code is built on open source "Mixed policy gradient" (see [here](https://github.com/idthanm/mpg.git)).
It is designed with TensorFlow 2 and Ray to realize a high-throughput asynchronous learning architecture, which modularizes the process of sampling, storing, learning, 
evaluating and testing with clear interfaces, organizes each of them in parallel. 


# Get started
Go to the train_scripts folder, wherein you can run two-state-mg.py or train_script.py to train
two different tasks.