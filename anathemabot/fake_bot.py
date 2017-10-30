"""
Serves as the connection between the instance of the neural network and the stdin/stdout of the Halite executable.
This script will be run for each player by the Halite executable, connecting to the network via named pipe and managing
the input and output for this specific player.
"""
import random
import os, sys
import uuid

NAMED_PIPE = "anathemapipe"
PLAYER_ID = uuid.uuid4()

with open(NAMED_PIPE) as fifo:
    # halite exe -> us -> anathema -> us -> halite exe

    # for each line from halite executable
    for line_from_halite in sys.stdin:

        fifo.write(line_from_halite) # write to anathema

        # if the data is not ordered correctly, may be impossible to get the frame that we want here.
        # may need to use sockets :(
        # is data line based??
        from_net = fifo.readline() # read from anathema

        # write to halite executable
        print(from_net)
