"""
Anathema: self-play reinforcement learning via a convolutional neural network
"""
import hlt
import manager_constants

from anathema import net as anet

import logging
import math, random
import numpy
import subprocess
import sys
import torch
import platform, os

NUM_PLAYERS = 2
NUM_GAMES = 1000

NUM_FEATURES = 7
NUM_OUTPUT_FEATURES = 3
HAS_CUDA = torch.cuda.is_available() and (platform.system() != 'Windows')

if HAS_CUDA:
    torch.cuda.device(0)

logging.info((platform.system()))

def clamp(value, small, large):
    return max(min(value, large), small)

def convert_map_to_tensor(game_map, input_tensor, my_ships):
    my_ships.clear()

    # feature vector: [ship hp, ship friendliness, docking status, planet hp, planet size, % docked_ships, planet friendliness]
    for player in game_map.all_players():
        owner_feature = 0 if player.id == game_map.my_id else 1
        for ship in player.all_ships():
            x = int(ship.x)
            y = int(ship.y)
            # hp from [0, 1]
            input_tensor[0][0][x][y] = ship.health / 255
            # friendless: 0 if me, 1 if enemy
            input_tensor[0][1][x][y] = owner_feature
            # 0 if undocked, .33 if docked, .66 if docking, 1 if undocking
            input_tensor[0][2][x][y] = ship.docking_status.value / 3

            if owner_feature == 0:
                my_ships[(x, y)] = ship

    for planet in game_map.all_planets():
        x = int(planet.x)
        y = int(planet.y)
        # hp from [0, 1]
        input_tensor[0][3][x][y] = planet.health / (planet.radius * 255)
        # radius from [0, 1]
        input_tensor[0][4][x][y] = (planet.radius - 3) / 5
        # % of docked ships [0, 1]
        input_tensor[0][5][x][y] = len(planet.all_docked_ships()) / planet.num_docking_spots
        # owner of this planet: -1 if me, 1 if enemy, 0 if unowned
        input_tensor[0][6][x][y] = (-1 if planet.owner == game_map.my_id else 1) if planet.is_owned() else 0

def one_or_negative_one():
    return 1 if random.random() > .5 else -1

def skew_towards_zero():
    return (1 - math.sqrt(math.sqrt(1 - random.random())))


def run_game(num_players, net):
    """
    Runs a single game against itself. Uses the same network to calculate moves for EACH player in this game.
    :param num_players: Number of players to simulate during this game (2-4)
    :return: n/a
    """
    # initialize halite
    run_commands = []
    for i in range(num_players):
        run_commands.append("./fake_bot {}".format(i))

    subprocess.Popen(["./halite", "-t", "-r", "-d", '60 40'] + run_commands)

    # GAME START
    games_per_player = []
    maps_per_player = []
    board_states_per_player = []
    outputs_per_player = []
    ships_per_player = []
    eliminated = []

    from_halite_fifos = []
    to_halite_fifos = []

    made_ships = False

    for i in range(num_players):
        from_halite_fifos.append(os.fdopen(os.open("pipes/from_halite_{}".format(i), os.O_RDONLY|os.O_NONBLOCK), "r"))
        to_halite_fifos.append(open("pipes/to_halite_{}".format(i), "w"))

        

        games_per_player.append(hlt.Game("Anathema", from_halite_fifos[i], to_halite_fifos[i]))
        logging.info("Starting << anathema >> for player {}".format(i))
        outputs_per_player.append([])
        board_states_per_player.append([])
        ships_per_player.append({})
        maps_per_player.append(None)
        eliminated.append(False)

    # Initialize zeroed input/output tensors
    input_tensor = torch.FloatTensor(1, NUM_FEATURES, games_per_player[0].map.width, games_per_player[0].map.height).zero_()
    output_tensor = torch.FloatTensor(1, NUM_OUTPUT_FEATURES, games_per_player[0].map.width, games_per_player[0].map.height).zero_()

    if HAS_CUDA:
        input_tensor = input_tensor.cuda()
        output_tensor = output_tensor.cuda()
        logging.info("Made it here")

    while True:

        # play out each player's turn
        for i, game in enumerate(games_per_player):

            if eliminated[i]:
                continue

            # need a way to detect when this player has lost and shouldnt be updated anymore
            try:

                #subprocess.call(["ps", "-ef", "|", "grep", "halite"])

                game_map = game.update_map()
            except ValueError as e:
                # this player is done playing
                logging.info(e)
                eliminated[i] = True

                from_halite_fifos[i].close()
                to_halite_fifos[i].close()

                if all(eliminated):
                    if made_ships:
                        return board_states_per_player[i], outputs_per_player[i]
                    else:
                        return [], []
                else:
                    continue

            command_queue = []
            my_ships = ships_per_player[i]
            if len(my_ships.keys()) > 3:
                made_ships = True
            input_tensor = torch.FloatTensor(1, NUM_FEATURES, games_per_player[0].map.width, games_per_player[0].map.height).zero_()
            output_tensor = torch.FloatTensor(1, NUM_OUTPUT_FEATURES, games_per_player[0].map.width, games_per_player[0].map.height).zero_()

            # Rebuild our input tensor based on the map state for this turn
            convert_map_to_tensor(game_map, input_tensor, my_ships)
            if len(my_ships.keys()) > 3:
                board_states_per_player[i].append(input_tensor[0])
                outputs_per_player[i].append(output_tensor[0])

            vi = torch.autograd.Variable(input_tensor)

            if HAS_CUDA:
                vi = vi.cuda()

            move_commands = net.forward(vi)[0].permute(1, 2, 0)

            for (x, y) in my_ships:
                this_ship = my_ships[(x, y)]
                angle, speed, dock = move_commands[x][y].data

                angle = (angle + (one_or_negative_one() * skew_towards_zero()))
                output_tensor[0][0][x][y] = angle
                command_angle = int(360 * angle) % 360

                speed = speed + (one_or_negative_one() * skew_towards_zero())
                output_tensor[0][1][x][y] = speed
                command_speed = numpy.clip(int(7 * speed), 0, 7)

                dock = numpy.clip(dock + (one_or_negative_one() * skew_towards_zero()), 0, 1)
                output_tensor[0][2][x][y] = dock
                command_dock = dock

                # Execute ship command
                if command_dock < 0.1:
                    # we want to undock
                    if this_ship.docking_status.value == this_ship.DockingStatus.DOCKED.value:
                        
                        output_tensor[0][2][x][y] = 1 - skew_towards_zero()
                        
                        #command_queue.append(this_ship.undock())
                    else:
                        command_queue.append(this_ship.thrust(command_speed, command_angle))
                else:
                    # we want to dock
                    if this_ship.docking_status.value == this_ship.DockingStatus.UNDOCKED.value:
                        closest_planet = this_ship.closest_planet(game_map)
                        if this_ship.can_dock(closest_planet):
                            command_queue.append(this_ship.dock(closest_planet))
                        else:
                            
                            command_queue.append(this_ship.thrust(command_speed, command_angle))

            # Send our set of commands to the Halite engine for this turn

            game.send_command_queue(command_queue)

def get_model_file_arg():

    if len(sys.argv) > 2:
        print("Too many arguments!")
        print("python TrainingBot.py [model file]")
        exit(-1)

    return sys.argv[1] if len(sys.argv) == 2 else None

def main():

    # load model from file
    model_file = get_model_file_arg()

    if model_file:
        net = torch.load(model_file)
    else:
        net = anet.Net()
        model_file = "AutoGeneratedModel-0"

    file_prefix, games_played = model_file.split("-")
    games_played = int(games_played)

    if HAS_CUDA:
        net.cuda()

    while True:

        states, outputs = [], []

        print("Game ID:", games_played)

        for game_id in range(0, manager_constants.rollout_games):

            tstates, toutputs = run_game(NUM_PLAYERS, net)

            states += tstates
            outputs += toutputs

            print("Training:", game_id)
            print('here', len(states), len(outputs))

            
            games_played += 1

        if len(states) > 0:
            net.my_train(torch.stack(states), torch.stack(outputs), epochs=1)
            print("Saving")
            torch.save(net, "{}-{}".format(file_prefix, games_played))

        
try:
    if __name__ == '__main__':
        try:
            main()
        except:
            logging.exception("Error in main program")
            
            raise
except:
    pass
finally:
    subprocess.call(["pkill", "fake"])
    subprocess.call(["pkill", "halite"])
    print("The end is nigh.")


