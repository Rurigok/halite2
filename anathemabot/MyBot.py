"""
Anathema: self-play reinforcement learning via a convolutional neural network
"""
from anathema import net as anet

import hlt
import logging
import torch
import random, math

NUM_FEATURES = 7
NUM_OUTPUT_FEATURES = 3
HAS_CUDA = torch.cuda.is_available()

def convert_map_to_tensor(game_map, input_tensor, my_ship_locations):

    my_ship_locations.clear()

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
                my_ship_locations[(x, y)] = ship

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

def distribution():
    return (1 - math.sqrt(1 - random.random()))

def main():
    # GAME START
    game = hlt.Game("Anathema")
    logging.info("Starting << anathema >>")

    # Initialize zeroed input tensor
    input_tensor = torch.FloatTensor(1, NUM_FEATURES, game.map.width, game.map.height).zero_()
    output_tensor = torch.FloatTensor(1, NUM_OUTPUT_FEATURES, game.map.width, game.map.height).zero_()

    if False and HAS_CUDA:
        input_tensor = input_tensor.cuda()

    net = anet.Net()

    game_history = []
    my_ship_locations = {}

    while True:
        # TURN START
        game_map = game.update_map()
        command_queue = []

        # Rebuild our input tensor based on the map state for this turn
        convert_map_to_tensor(game_map, input_tensor, my_ship_locations)
        #input_tensor = input_tensor.unsqueeze(0)
        vi = torch.autograd.Variable(input_tensor)
        move_commands = net.forward(vi)[0].permute(1, 2, 0)

        for (x, y) in my_ship_locations:
            this_ship = my_ship_locations[(x, y)]
            angle, speed, dock = move_commands[x][y].data

            angle = (angle + (one_or_negative_one() * distribution()))

            # Set angle of the output tensor to the skewed angle
            output_tensor[0][0][x][y] = angle

            command_angle = int(360 * angle)

            speed = speed + (one_or_negative_one() * distribution())

            # Set speed of the output tensor to skewed speed
            output_tensor[0][1][x][y] = speed

            command_speed = int(7 * speed)

            dock = dock + (one_or_negative_one() * distribution())

            # Set dock of the output tensor to skewed dock

            command_dock = dock

            # [0, .5) = undock
            # [.5 , 1] = dock
            if command_dock < .5:
                # we want to undock
                if this_ship.docking_status.value == this_ship.DockingStatus.DOCKED:
                    command_queue.append(this_ship.undock())
                else:
                    command_queue.append(this_ship.thrust(command_speed, command_angle))
            else:
                # we want to dock
                if this_ship.docking_status.value == this_ship.DockingStatus.UNDOCKED:
                    #command_queue.append(this_ship.dock()) HARD TO DO
                    pass
                else:
                    command_queue.append(this_ship.thrust(command_speed, command_angle))

        # Here we define the set of commands to be sent to the Halite engine at the end of the turn

        # For every ship that I control
        # for ship in game_map.get_me().all_ships():
        #     # If the ship is docked
        #     if ship.docking_status != ship.DockingStatus.UNDOCKED:
        #         # Skip this ship
        #         continue
        #
        #     # For each planet in the game (only non-destroyed planets are included)
        #     for planet in game_map.all_planets():
        #         # If the planet is owned
        #         if planet.is_owned():
        #             # Skip this planet
        #             continue
        #
        #         # If we can dock, let's (try to) dock. If two ships try to dock at once, neither will be able to.
        #         if ship.can_dock(planet):
        #             # We add the command by appending it to the command_queue
        #             command_queue.append(ship.dock(planet))
        #         else:
        #             # If we can't dock, we move towards the closest empty point near this planet (by using closest_point_to)
        #             # with constant speed. Don't worry about pathfinding for now, as the command will do it for you.
        #             # We run this navigate command each turn until we arrive to get the latest move.
        #             # Here we move at half our maximum speed to better control the ships
        #             # In order to execute faster we also choose to ignore ship collision calculations during navigation.
        #             # This will mean that you have a higher probability of crashing into ships, but it also means you will
        #             # make move decisions much quicker. As your skill progresses and your moves turn more optimal you may
        #             # wish to turn that option off.
        #             navigate_command = ship.navigate(ship.closest_point_to(planet), game_map,
        #                                              speed=hlt.constants.MAX_SPEED / 2, ignore_ships=True)
        #             # If the move is possible, add it to the command_queue (if there are too many obstacles on the way
        #             # or we are trapped (or we reached our destination!), navigate_command will return null;
        #             # don't fret though, we can run the command again the next turn)
        #             if navigate_command:
        #                 command_queue.append(navigate_command)
        #         break

        # Send our set of commands to the Halite engine for this turn
        game.send_command_queue(command_queue)
        # TURN END
        # GAME END


if __name__ == '__main__':
    try:
        main()
    except:
        logging.exception("Error in main program")
        raise
