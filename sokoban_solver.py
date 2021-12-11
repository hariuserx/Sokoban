# Preprocess the data

import random
import time
from collections import defaultdict

start_time = time.time()
random.seed(0)

with open("./benchmarks-fixed/sokoban-04.txt", "r") as file:
    dimensions = file.readline()
    row_size = int(dimensions.split()[0])
    col_size = int(dimensions.split()[1])


    def construct_pairs(n, coordinates):
        ans = set()
        for ix in range(1, n * 2, 2):
            ans.add((int(coordinates[ix]), int(coordinates[ix + 1])))
        return ans


    # Walls
    walls_line = file.readline().split()
    n_walls = int(len(walls_line) / 2)
    wall_coordinates = construct_pairs(n_walls, walls_line)
    print("Wall coordinates", wall_coordinates)

    # Boxes
    boxes_line = file.readline().split()
    n_boxes = int(len(boxes_line) / 2)
    print(n_boxes)
    print(boxes_line)
    box_coordinates = construct_pairs(n_boxes, boxes_line)

    # Storage
    storage_line = file.readline().split()
    n_storage = int(len(storage_line) / 2)
    storage_coordinates = construct_pairs(n_storage, storage_line)

    player_location_line = file.readline().split()
    initial_player_location = (int(player_location_line[0]), int(player_location_line[1]))

directions = ["U", "D", "L", "R"]
discount = 1.0
learning_rate = 0.7


# Utility methods
def manhattan_distance(coordinates1, coordinates2):
    return abs(coordinates1[0] - coordinates2[0]) + abs(coordinates1[1] - coordinates2[1])


def closeness_heuristic(boxes, storages):
    ans = 0
    boxes_not_in_storage = boxes.difference(storages)
    empty_storages = storages.difference(boxes)

    for box in boxes_not_in_storage:
        minx = float('inf')
        min_storage = None
        for storage in empty_storages:
            distance = manhattan_distance(box, storage)
            if distance < minx:
                minx = distance
                min_storage = storage
        ans += minx
        empty_storages.remove(min_storage)

    return ans


def is_deadlock(box):
    if box in storage_coordinates:
        return False
    for adjacent in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]:
        if (box[0] + adjacent[0], box[1]) in wall_coordinates and (box[0], box[1] + adjacent[1]) in wall_coordinates:
            return True
    return False


def agent_to_box_min_distance(box_coordinates_inp, player_location):
    minx = float("inf")
    for box in box_coordinates_inp.difference(storage_coordinates):
        minx = min(manhattan_distance(player_location, box), minx)
    return minx


def get_next_player_and_box_location(action, current_location):
    player_next_location = None
    box_next_location_if_present = None
    if action == 'U':
        player_next_location = (current_location[0] - 1, current_location[1])
        box_next_location_if_present = (current_location[0] - 2, current_location[1])
    elif action == 'D':
        player_next_location = (current_location[0] + 1, current_location[1])
        box_next_location_if_present = (current_location[0] + 2, current_location[1])
    elif action == 'R':
        player_next_location = (current_location[0], current_location[1] + 1)
        box_next_location_if_present = (current_location[0], current_location[1] + 2)
    elif action == 'L':
        player_next_location = (current_location[0], current_location[1] - 1)
        box_next_location_if_present = (current_location[0], current_location[1] - 2)
    return player_next_location, box_next_location_if_present


def check_if_move_is_invalid(player_location, action, box_coordinates_inp):
    player_next_location, box_next_location_if_present = get_next_player_and_box_location(action, player_location)
    return (player_next_location in wall_coordinates) \
           or (player_next_location in box_coordinates_inp and box_next_location_if_present in box_coordinates_inp) \
           or (player_next_location in box_coordinates_inp and box_next_location_if_present in wall_coordinates)


def get_all_possible_actions(player_location, box_coordinates_inp):
    return [action for action in directions if not check_if_move_is_invalid(player_location, action,
                                                                            box_coordinates_inp)]


def get_state_for_action(player_location, action, box_coordinates_inp):
    targetLocation, targetNextLocation = get_next_player_and_box_location(action, player_location)
    newLocation = targetLocation
    newBoxCoordinates = set(box_coordinates_inp.copy())
    if targetLocation in newBoxCoordinates:
        newBoxCoordinates.remove(targetLocation)
        newBoxCoordinates.add(targetNextLocation)
    return BoardState(newBoxCoordinates, newLocation)


# Sokoban state
class BoardState:
    def __init__(self, box_coordinates_inp, player_location):
        self.box_coordinates = frozenset(box_coordinates_inp)
        self.player_location = player_location

    def __eq__(self, other):
        if type(other) is type(self):
            return (self.box_coordinates, self.player_location) == (other.box_coordinates, other.player_location)
        return False

    def __hash__(self):
        return hash((self.box_coordinates, self.player_location))

    def __repr__(self):
        return str(self.box_coordinates) + " " + str(self.player_location)

    def boxes_not_in_destination(self):
        return len(self.box_coordinates.difference(storage_coordinates))

    def goal_reached(self):
        return len(storage_coordinates.difference(self.box_coordinates)) == 0

    def is_stuck(self):
        return any(map(lambda box: is_deadlock(box), self.box_coordinates))


class SokobanBoard:
    # static
    qtable = {}

    def __init__(self, box_coordinates_inp, location):
        self.current_state = BoardState(box_coordinates_inp, location)

    def clear_qtable(self):
        self.__class__.qtable.clear()

    def get_current_state_actions(self):
        return get_all_possible_actions(self.current_state.player_location, self.current_state.box_coordinates)

    def set_QValue(self, state, action, newValue):
        self.__class__.qtable[(state, action)] = newValue

    def get_QValue(self, state, action):
        if state.goal_reached():
            self.set_QValue(state, action, float("inf"))
            return float("inf")
        if state.is_stuck():
            self.set_QValue(state, action, float("-inf"))
            return float("-inf")
        return self.__class__.qtable.get((state, action), 0)


def get_max_QValue(sokoban_board, state, possible_actions):
    max_qvalue = float("-inf")
    max_action = None
    for action in possible_actions:
        if max_qvalue <= sokoban_board.get_QValue(state, action):
            max_qvalue = sokoban_board.get_QValue(state, action)
            max_action = action

    return [max_qvalue, max_action]


def get_action_with_highest_qvalue(sokoban_board):
    possible_actions = get_all_possible_actions(sokoban_board.current_state.player_location,
                                                sokoban_board.current_state.box_coordinates)
    return get_max_QValue(sokoban_board, sokoban_board.current_state, possible_actions)[1]


def perform_valid_action(sokoban_board, action):
    new_state = get_state_for_action(sokoban_board.current_state.player_location, action,
                                     sokoban_board.current_state.box_coordinates)

    R = -1

    # Boxes closeness to the storage locations
    step_difference = closeness_heuristic(new_state.box_coordinates, storage_coordinates) - \
                      closeness_heuristic(sokoban_board.current_state.box_coordinates, storage_coordinates)
    if step_difference < 0:
        R += 3
    elif step_difference > 0:
        R += -3

    # Move a box to storage
    remaining_boxes_difference = new_state.boxes_not_in_destination() - \
                                 sokoban_board.current_state.boxes_not_in_destination()
    if remaining_boxes_difference < 0:
        R += 15
    elif remaining_boxes_difference > 0:
        R += -10

    # Player is close to box
    distance_to_closest_box_difference = agent_to_box_min_distance(new_state.box_coordinates,
                                                                   new_state.player_location) \
                                         - agent_to_box_min_distance(sokoban_board.current_state.box_coordinates,
                                                                     sokoban_board.current_state.player_location)
    if distance_to_closest_box_difference < 0:
        R += 1
    elif distance_to_closest_box_difference > 0:
        R += -1

    current_q_value = sokoban_board.get_QValue(sokoban_board.current_state, action)
    new_q_value = current_q_value + learning_rate * (
            R + discount * get_max_QValue(sokoban_board, new_state,
                                          get_all_possible_actions(new_state.player_location,
                                                                   new_state.box_coordinates))[0]
            - current_q_value)
    sokoban_board.set_QValue(sokoban_board.current_state, action, new_q_value)
    sokoban_board.current_state = new_state


# BFS part. Disconnected as its not performing well. May perform well for very large boards
class SokobanGraphRepresentaition:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def BFS(self, s, d):
        if s == d:
            return True

        visited = [False] * (len(self.graph) + 1)

        queue = [s]

        visited[s] = True
        while queue:

            s = queue.pop(0)

            for idx in self.graph[s]:
                if idx == d:
                    return True

                if not visited[idx]:
                    queue.append(idx)
                    visited[idx] = True

        return False


def isSafe(idx, j, matrix):
    if 0 <= idx <= len(matrix) and 0 <= j <= len(matrix[0]):
        return True
    else:
        return False


def findPath(sokoban_board, box_location_adjacent):
    g = SokobanGraphRepresentaition()
    k = 1
    for idx in range(row_size):
        for j in range(col_size):
            if (idx, j) in sokoban_board[idx][j] != 0:
                if isSafe(idx, j + 1, sokoban_board):
                    g.add_edge(k, k + 1)
                if isSafe(idx, j - 1, sokoban_board):
                    g.add_edge(k, k - 1)
                if isSafe(idx + 1, j, sokoban_board):
                    g.add_edge(k, k + col_size)
                if isSafe(idx - 1, j, sokoban_board):
                    g.add_edge(k, k - col_size)
            k += 1

    return g.BFS(sokoban_board.current_state.player_location, box_location_adjacent)


def run_an_episode_and_check_if_terminal(player_start_location_inp, max_moves_inp, epsilon_inp):
    # Print path if this episode is terminal. Return true if terminal
    sokoban_board = SokobanBoard(box_coordinates, player_start_location_inp)
    path = ""
    for _ in range(max_moves_inp):
        if random.random() < epsilon_inp:
            action = random.choice(get_all_possible_actions(sokoban_board.current_state.player_location,
                                                            sokoban_board.current_state.box_coordinates))
            path += action
        else:
            action = get_action_with_highest_qvalue(sokoban_board)
            path += action
        perform_valid_action(sokoban_board, action)
        if sokoban_board.current_state.goal_reached():
            print("Number of moves ", len(path))
            print("Path is %%%%%%% ", path)
            return True
    return False


max_episodes = row_size * col_size * n_storage * 1000
max_max_moves = row_size * col_size * n_storage * 1
min_max_moves = row_size * col_size * n_storage * 1
epsilon_min = 0.2  # Less exploration after some episodes
epsilon_max = 0.2  # explore lot in the beginning
print('Total episodes: ', max_episodes)
# Run the algorithm
for i in range(max_episodes):
    episodeReachedTerminal = run_an_episode_and_check_if_terminal(initial_player_location, max_max_moves, epsilon_max)
    if episodeReachedTerminal:
        break
    if i % 1000 == 0:
        max_max_moves = max(int(max_max_moves * 0.9), min_max_moves)
        epsilon_max = max(epsilon_max * 0.9, epsilon_min)
        print("Max moves: ", max_max_moves)
        print("current epsilon: ", epsilon_max)
        print("Total Completed Episodes: ", i)
        print("Qtable size: ", len(SokobanBoard.qtable))
        timeElapsed = time.time() - start_time
        print('Time elapsed: ', timeElapsed)
        if timeElapsed > 3600:
            print('Timeout')
            break

end = time.time()
print('Total time for solution: ', end - start_time)
