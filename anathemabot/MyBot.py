"""
Welcome to your first Halite-II bot!

This bot's name is Settler. It's purpose is simple (don't expect it to win complex games :) ):
1. Initialize game
2. If a ship is not docked and there are unowned planets
2.a. Try to Dock in the planet if close enough
2.b If not, go towards the planet

Note: Please do not place print statements here as they are used to communicate with the Halite engine. If you need
to log anything use the logging module.
"""
# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
# Then let's import the logging module so we can print out information
import logging
import torch

def convert_map_to_tensor(game_map, input_tensor):

    # feature vector: [ship hp, ship friendliness, planet hp, planet size, % docked_ships, planet friendliness]
    for player in game_map.all_players():
        for ship in player.all_ships():
            x = int(ship.x)
            y = int(ship.y)
            input_tensor[0][x][y] = ship.health / 255
            input_tensor[1][x][y] = 0 if ship.owner == game_map.my_id else 1

    for planet in game_map.all_planets():
        x = int(planet.x)
        y = int(planet.y)
        input_tensor[2][x][y] = planet.health / (planet.radius * 255) # hp from [0, 1]
        input_tensor[3][x][y] = (planet.radius - 3) / 5 # radius from [0, 1]
        input_tensor[4][x][y] = len(planet.all_docked_ships()) / planet.num_docking_spots
        input_tensor[5][x][y] = (-1 if planet.owner == game_map.my_id else 1) if planet.is_owned() else 0

    logging.info(input_tensor)

# GAME START
game = hlt.Game("Anathema")
logging.info("Starting << anathema >>")

# Initialize tensor
input_tensor = torch.FloatTensor(4, game.map.width, game.map.height).zero_()

while True:
    # TURN START
    # Update the map for the new turn and get the latest version
    game_map = game.update_map()

    # Build our input tensor based on the map state for this turn
    convert_map_to_tensor(game_map, input_tensor)

    # Here we define the set of commands to be sent to the Halite engine at the end of the turn
    command_queue = []
    # For every ship that I control
    for ship in game_map.get_me().all_ships():
        # If the ship is docked
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            # Skip this ship
            continue

        # For each planet in the game (only non-destroyed planets are included)
        for planet in game_map.all_planets():
            # If the planet is owned
            if planet.is_owned():
                # Skip this planet
                continue

            # If we can dock, let's (try to) dock. If two ships try to dock at once, neither will be able to.
            if ship.can_dock(planet):
                # We add the command by appending it to the command_queue
                command_queue.append(ship.dock(planet))
            else:
                # If we can't dock, we move towards the closest empty point near this planet (by using closest_point_to)
                # with constant speed. Don't worry about pathfinding for now, as the command will do it for you.
                # We run this navigate command each turn until we arrive to get the latest move.
                # Here we move at half our maximum speed to better control the ships
                # In order to execute faster we also choose to ignore ship collision calculations during navigation.
                # This will mean that you have a higher probability of crashing into ships, but it also means you will
                # make move decisions much quicker. As your skill progresses and your moves turn more optimal you may
                # wish to turn that option off.
                navigate_command = ship.navigate(ship.closest_point_to(planet), game_map, speed=hlt.constants.MAX_SPEED/2, ignore_ships=True)
                # If the move is possible, add it to the command_queue (if there are too many obstacles on the way
                # or we are trapped (or we reached our destination!), navigate_command will return null;
                # don't fret though, we can run the command again the next turn)
                if navigate_command:
                    command_queue.append(navigate_command)
            break

    # Send our set of commands to the Halite engine for this turn
    game.send_command_queue(command_queue)
    # TURN END
# GAME END
