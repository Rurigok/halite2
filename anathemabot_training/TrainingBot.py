"""
Anathema: self-play reinforcement learning via a convolutional neural network
"""
from anathema import net as anet

import hlt
import logging
import math, random
import numpy
import subprocess
import torch
import platform

NUM_PLAYERS = 2
NUM_GAMES = 1000

NUM_FEATURES = 7
NUM_OUTPUT_FEATURES = 3
HAS_CUDA = torch.cuda.is_available() and (platform.system() != 'Windows')
logging.info((platform.system()))

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
    return (1 - math.sqrt(1 - random.random()))


def run_game(num_players, net):
    """
    Runs a single game against itself. Uses the same network to calculate moves for EACH player in this game.
    :param num_players: Number of players to simulate during this game (2-4)
    :return: n/a
    """
    # initialize halite
    subprocess.Popen(["./halite", "-t", "-d", '240 160'] + (['./fake_bot'] * num_players))

    # GAME START
    games_per_player = []
    maps_per_player = []
    outputs_per_player = []
    ships_per_player = []
    eliminated = []

    from_halite_fifos = []
    to_halite_fifos = []

    for i in range(num_players):
        from_halite_fifos[i] = open("pipes/from_halite_{}".format(i), "r")
        to_halite_fifos[i] = open("pipes/to_halite_{}".format(i), "w")

        games_per_player.append(hlt.Game("Anathema", from_halite_fifos[i], to_halite_fifos[i]))
        logging.info("Starting << anathema >> for player {}".format(i))
        outputs_per_player.append([])
        ships_per_player.append({})
        maps_per_player.append(None)
        eliminated.append(False)

    # Initialize zeroed input/output tensors
    input_tensor = torch.FloatTensor(1, NUM_FEATURES, games_per_player[0].width, games_per_player[0].height).zero_()
    output_tensor = torch.FloatTensor(1, NUM_OUTPUT_FEATURES, games_per_player[0].width, games_per_player[0].height).zero_()

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
                game_map = game.update_map()
            except ValueError as e:
                # this player is done playing
                logging.info(e)
                eliminated[i] = True

                from_halite_fifos[i].close()
                to_halite_fifos[i].close()

                if all(eliminated):
                    return outputs_per_player[i]

            command_queue = []
            my_ships = ships_per_player[i]

            # Rebuild our input tensor based on the map state for this turn
            convert_map_to_tensor(game_map, input_tensor, my_ships)
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

                dock = dock + (one_or_negative_one() * skew_towards_zero())
                output_tensor[0][2][x][y] = dock
                command_dock = dock

                outputs_per_player[i].append(output_tensor)

                # Execute ship command
                if command_dock < .5:
                    # we want to undock
                    if this_ship.docking_status.value == this_ship.DockingStatus.DOCKED:
                        command_queue.append(this_ship.undock())
                    else:
                        command_queue.append(this_ship.thrust(command_speed, command_angle))
                else:
                    # we want to dock
                    if this_ship.docking_status.value == this_ship.DockingStatus.UNDOCKED:
                        closest_planet = this_ship.closest_planet(game_map)
                        if this_ship.can_dock(closest_planet):
                            command_queue.append(this_ship.dock(closest_planet))
                    else:
                        command_queue.append(this_ship.thrust(command_speed, command_angle))

            # Send our set of commands to the Halite engine for this turn
            game.send_command_queue(command_queue)

def main():
    # load model from file
    net = anet.Net()

    winning_outputs = run_game(NUM_PLAYERS, net)

if __name__ == '__main__':
    try:
        main()
    except:
        logging.exception("Error in main program")
        raise