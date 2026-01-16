from collections import deque
from heapq import heappush, heappop
from PIL import Image, ImageDraw
import os

CELL_SIZE = 40
MAZE_FILE = "maze4.txt"
OUT_IMAGE = "maze.png"
class Node:
    def __init__(self, state, parent, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

class Frontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        raise NotImplementedError

    def remove(self):
        raise NotImplementedError

    def empty(self):
        return len(self.frontier) == 0

    def contains_state(self, state):
        for item in self.frontier:
            if isinstance(item, tuple):
                node = item[2] if len(item) == 3 else item[1]
            else:
                node = item
            if node.state == state:
                return True
        return False



class StackFrontier(Frontier):
    def add(self, node):
        self.frontier.append(node)

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        return self.frontier.pop()


class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        return self.frontier.pop(0)


class GreedyFrontier(Frontier):
    def __init__(self, goal):
        super().__init__()
        self.goal = goal
        self.counter = 0

    def heuristic(self, state):
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

    def add(self, node):
        heappush(self.frontier, (self.heuristic(node.state), self.counter, node))
        self.counter += 1

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        return heappop(self.frontier)[2]


class AStarFrontier(GreedyFrontier):
    def __init__(self, goal):
        super().__init__(goal)
        self.counter = 0
    
    def add(self, node):
        f = node.cost + self.heuristic(node.state) # h(n) is cost to reach goal from current node and g(n) is the cost to reach that neighbor node 
        heappush(self.frontier, (f, self.counter, node))
        self.counter += 1
    
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        return heappop(self.frontier)[2]


class Maze:
    def __init__(self, filename):
        with open(filename) as f:
            self.grid = [list(line.rstrip("\n")) for line in f]

        self.height = len(self.grid)
        self.width = max(len(row) for row in self.grid) if self.grid else 0
        self.start, self.goal = self.find_points()

    def find_points(self):
        for y in range(self.height):
            for x in range(len(self.grid[y])):  
                if self.grid[y][x] == "A":
                    start = (x, y)
                elif self.grid[y][x] == "B":
                    goal = (x, y)
        return start, goal

    def neighbors(self, state):
        x, y = state
        candidates = [
            ("right", (x+1, y)),
            ("left", (x-1, y)),
            ("down", (x, y+1)),
            ("up", (x, y-1))
        ]

        result = []
        for action, (nx, ny) in candidates:
            if 0 <= ny < self.height and 0 <= nx < len(self.grid[ny]):
                if self.grid[ny][nx] != "#":
                    result.append((action, (nx, ny)))
        return result


class MazeSolver:
    def __init__(self, maze, frontier):
        self.maze = maze
        self.frontier = frontier
        self.explored = set()

    def solve(self):
        start = Node(self.maze.start, None, None, 0)
        self.frontier.add(start)

        while True:
            if self.frontier.empty():
                return []

            node = self.frontier.remove()

            if node.state == self.maze.goal:
                return self.reconstruct(node)

            self.explored.add(node.state)

            for action, state in self.maze.neighbors(node.state):
                if state not in self.explored and not self.frontier.contains_state(state):
                    child = Node(
                        state=state,
                        parent=node,
                        action=action,
                        cost=node.cost + 1
                    )
                    self.frontier.add(child)

    def reconstruct(self, node):
        path = []
        while node.parent is not None:
            path.append(node.state)
            node = node.parent
        path.append(self.maze.start)
        return path[::-1]


# def draw_maze(maze, path, explored):
#     img = Image.new("RGB", (maze.width*CELL_SIZE, maze.height*CELL_SIZE), "white")
#     px = img.load()

#     COLORS = {
#         "wall": (0,0,0),
#         "empty": (255,255,255),
#         "explored": (255,165,0),
#         "path": (255,255,0),
#         "start": (255,0,0),
#         "goal": (0,255,0)
#     }

#     for y in range(maze.height):
#         for x in range(len(maze.grid[y])):
#             color = COLORS["empty"]
#             if maze.grid[y][x] == "#":
#                 color = COLORS["wall"]
#             elif (x,y) in explored:
#                 color = COLORS["explored"]

#             for i in range(CELL_SIZE):
#                 for j in range(CELL_SIZE):
#                     px[x*CELL_SIZE+i, y*CELL_SIZE+j] = color

#     for x,y in path:
#         for i in range(CELL_SIZE):
#             for j in range(CELL_SIZE):
#                 px[x*CELL_SIZE+i, y*CELL_SIZE+j] = COLORS["path"]

#     sx, sy = maze.start
#     gx, gy = maze.goal

#     for i in range(CELL_SIZE):
#         for j in range(CELL_SIZE):
#             px[sx*CELL_SIZE+i, sy*CELL_SIZE+j] = COLORS["start"]
#             px[gx*CELL_SIZE+i, gy*CELL_SIZE+j] = COLORS["goal"]

#     img.save(OUT_IMAGE)
def draw_maze(maze, path, explored):
    global OUT_IMAGE
    # Initialize image and drawing context
    img = Image.new("RGB", (maze.width * CELL_SIZE, maze.height * CELL_SIZE), "black")
    draw = ImageDraw.Draw(img)

    COLORS = {
        "wall": (33, 37, 41),      # Dark grey/black for walls
        "empty": (255, 255, 255),  # White for unvisited
        "explored": (212, 97, 85), # Reddish-orange for explored
        "path": (255, 255, 102),   # Yellow for the solution path
        "start": (0, 171, 28),     # Green for start
        "goal": (255, 0, 0)        # Red for goal
    }

    for y in range(maze.height):
        for x in range(len(maze.grid[y])):
            # Default to white
            fill = COLORS["empty"]

            # Determine cell color
            if maze.grid[y][x] == "#":
                fill = COLORS["wall"]
            elif (x, y) == maze.start:
                fill = COLORS["start"]
            elif (x, y) == maze.goal:
                fill = COLORS["goal"]
            elif (x, y) in path:
                fill = COLORS["path"]
            elif (x, y) in explored:
                fill = COLORS["explored"]

            # Draw the cell with a 1-pixel black border
            draw.rectangle(
                ([(x * CELL_SIZE, y * CELL_SIZE), 
                  ((x + 1) * CELL_SIZE - 1, (y + 1) * CELL_SIZE - 1)]),
                fill=fill,
                outline="black"
            )

    img.save(OUT_IMAGE)

def print_solution(maze, path):
    for x,y in path:
        if maze.grid[y][x] not in ("A","B"):
            maze.grid[y][x] = "*"
    for row in maze.grid:
        print("".join(row))

def main():
    global OUT_IMAGE
    maze = Maze(MAZE_FILE)
    frontier = StackFrontier()        # DFS
    # frontier = QueueFrontier()        # BFS
    # frontier = GreedyFrontier(maze.goal)
    # frontier = AStarFrontier(maze.goal)
    OUT_IMAGE = "maze-dfs.png"
    solver = MazeSolver(maze, frontier)
    path = solver.solve()

    print_solution(maze, path)
    draw_maze(maze, path, solver.explored)

    print("\nSolved using:", frontier.__class__.__name__)
    print("Path length:", len(path))


    frontier = QueueFrontier()  
    OUT_IMAGE = "maze-bfs.png"
    solver = MazeSolver(maze, frontier)
    path = solver.solve()

    print_solution(maze, path)
    draw_maze(maze, path, solver.explored)

    print("\nSolved using:", frontier.__class__.__name__)
    print("Path length:", len(path))


    frontier = GreedyFrontier(maze.goal)
    OUT_IMAGE = "maze-greed.png"
    solver = MazeSolver(maze, frontier)
    path = solver.solve()

    print_solution(maze, path)
    draw_maze(maze, path, solver.explored)

    print("\nSolved using:", frontier.__class__.__name__)
    print("Path length:", len(path))


    frontier = AStarFrontier(maze.goal)
    OUT_IMAGE = "maze-a*.png"
    solver = MazeSolver(maze, frontier)
    path = solver.solve()

    print_solution(maze, path)
    draw_maze(maze, path, solver.explored)

    print("\nSolved using:", frontier.__class__.__name__)
    print("Path length:", len(path))

if __name__ == "__main__":
    main()