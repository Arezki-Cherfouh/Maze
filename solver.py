import cv2
import numpy as np
import pyautogui
import time
from heapq import heappush, heappop
from PIL import Image, ImageGrab
import tkinter as tk
from tkinter import messagebox, filedialog

# Disable pyautogui failsafe for smooth automation
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


class Node:
    """Node for A* pathfinding"""
    def __init__(self, state, parent, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost


class AStarFrontier:
    """A* frontier using priority queue"""
    def __init__(self, goal):
        self.frontier = []
        self.goal = goal
        self.counter = 0

    def heuristic(self, state):
        """Manhattan distance heuristic"""
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

    def add(self, node):
        f = node.cost + self.heuristic(node.state)
        heappush(self.frontier, (f, self.counter, node))
        self.counter += 1

    def remove(self):
        if self.empty():
            raise Exception("Empty frontier")
        return heappop(self.frontier)[2]

    def empty(self):
        return len(self.frontier) == 0

    def contains_state(self, state):
        for item in self.frontier:
            node = item[2]
            if node.state == state:
                return True
        return False


class MazeDetector:
    """Detects and processes maze from image"""
    
    def __init__(self, image):
        if isinstance(image, str):
            self.original = cv2.imread(image)
        else:
            self.original = np.array(image)
            self.original = cv2.cvtColor(self.original, cv2.COLOR_RGB2BGR)
        
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.grid = None
        self.start = None
        self.goal = None
        self.cell_size = None
        
    def detect_maze(self):
        """Detect maze structure from image"""
        # Apply threshold to get binary image
        _, binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours to detect maze boundaries
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("No maze detected in image")
        
        # Get the largest contour (assumed to be the maze)
        maze_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(maze_contour)
        
        # Crop to maze area
        maze_region = binary[y:y+h, x:x+w]
        
        # Estimate cell size by finding grid lines
        self.cell_size = self._estimate_cell_size(maze_region)
        
        if self.cell_size < 5:
            self.cell_size = 20  # Default fallback
        
        # Create grid representation
        self.grid = self._create_grid(maze_region)
        
        # Detect start and goal
        self._detect_start_goal()
        
        return self.grid, self.start, self.goal
    
    def _estimate_cell_size(self, image):
        """Estimate cell size from grid lines"""
        height, width = image.shape
        
        # Sample horizontal lines
        mid_row = height // 2
        row_data = image[mid_row, :]
        
        # Find transitions (black to white or vice versa)
        transitions = []
        for i in range(1, len(row_data)):
            if row_data[i] != row_data[i-1]:
                transitions.append(i)
        
        if len(transitions) > 1:
            # Calculate average distance between transitions
            distances = [transitions[i+1] - transitions[i] for i in range(len(transitions)-1)]
            if distances:
                avg_cell_size = np.median(distances)
                return int(avg_cell_size)
        
        return 20  # Default
    
    def _create_grid(self, image):
        """Convert image to grid representation"""
        height, width = image.shape
        
        grid_height = height // self.cell_size
        grid_width = width // self.cell_size
        
        grid = []
        for row in range(grid_height):
            grid_row = []
            for col in range(grid_width):
                # Sample center of cell
                y = row * self.cell_size + self.cell_size // 2
                x = col * self.cell_size + self.cell_size // 2
                
                if y < height and x < width:
                    pixel_value = image[y, x]
                    # Black = wall, White = path
                    if pixel_value < 128:
                        grid_row.append('#')
                    else:
                        grid_row.append(' ')
                else:
                    grid_row.append('#')
            grid.append(grid_row)
        
        return grid
    
    def _detect_start_goal(self):
        """Detect start (usually green) and goal (usually red) positions"""
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        
        # Green range for start (adjust as needed)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Red range for goal (adjust as needed)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Find start position
        green_points = cv2.findNonZero(green_mask)
        if green_points is not None:
            avg_point = np.mean(green_points, axis=0)[0]
            grid_x = int(avg_point[0] // self.cell_size)
            grid_y = int(avg_point[1] // self.cell_size)
            self.start = (grid_x, grid_y)
            if 0 <= grid_y < len(self.grid) and 0 <= grid_x < len(self.grid[0]):
                self.grid[grid_y][grid_x] = 'A'
        
        # Find goal position
        red_points = cv2.findNonZero(red_mask)
        if red_points is not None:
            avg_point = np.mean(red_points, axis=0)[0]
            grid_x = int(avg_point[0] // self.cell_size)
            grid_y = int(avg_point[1] // self.cell_size)
            self.goal = (grid_x, grid_y)
            if 0 <= grid_y < len(self.grid) and 0 <= grid_x < len(self.grid[0]):
                self.grid[grid_y][grid_x] = 'B'
        
        # If start/goal not detected by color, find first/last empty cells
        if self.start is None or self.goal is None:
            self._find_start_goal_by_position()
    
    def _find_start_goal_by_position(self):
        """Find start and goal by position (top-left and bottom-right empty cells)"""
        if self.start is None:
            for y in range(len(self.grid)):
                for x in range(len(self.grid[0])):
                    if self.grid[y][x] == ' ':
                        self.start = (x, y)
                        self.grid[y][x] = 'A'
                        break
                if self.start:
                    break
        
        if self.goal is None:
            for y in range(len(self.grid)-1, -1, -1):
                for x in range(len(self.grid[0])-1, -1, -1):
                    if self.grid[y][x] == ' ':
                        self.goal = (x, y)
                        self.grid[y][x] = 'B'
                        break
                if self.goal:
                    break


class MazeSolver:
    """Solves maze using A* algorithm"""
    
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        self.explored = set()
        
    def neighbors(self, state):
        """Get valid neighbors of current state"""
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
                if self.grid[ny][nx] != '#':
                    result.append((action, (nx, ny)))
        return result
    
    def solve(self):
        """Solve maze using A* algorithm"""
        frontier = AStarFrontier(self.goal)
        start_node = Node(self.start, None, None, 0)
        frontier.add(start_node)
        
        while not frontier.empty():
            node = frontier.remove()
            
            if node.state == self.goal:
                return self._reconstruct_path(node)
            
            self.explored.add(node.state)
            
            for action, state in self.neighbors(node.state):
                if state not in self.explored and not frontier.contains_state(state):
                    child = Node(
                        state=state,
                        parent=node,
                        action=action,
                        cost=node.cost + 1
                    )
                    frontier.add(child)
        
        return []  # No solution found
    
    def _reconstruct_path(self, node):
        """Reconstruct path from goal to start"""
        actions = []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return actions[::-1]


class MazeController:
    """Controls maze navigation using keyboard"""
    
    def __init__(self, actions, delay=1.0):
        self.actions = actions
        self.delay = delay
        
    def execute(self):
        """Execute the solution by pressing arrow keys"""
        print(f"\nStarting maze navigation in 3 seconds...")
        print(f"Total moves: {len(self.actions)}")
        print("Move your mouse to top-left corner to emergency stop!")
        
        time.sleep(3)
        
        for i, action in enumerate(self.actions):
            print(f"Move {i+1}/{len(self.actions)}: {action}")
            
            if action == "up":
                pyautogui.press('up')
            elif action == "down":
                pyautogui.press('down')
            elif action == "left":
                pyautogui.press('left')
            elif action == "right":
                pyautogui.press('right')
            
            time.sleep(self.delay)
        
        print("\n✓ Maze solved successfully!")


class RegionSelector:
    """Interactive region selector for screen capture"""
    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect = None
        
    def select_region(self):
        """Open fullscreen window to select region"""
        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)
        root.configure(bg='black')
        
        canvas = tk.Canvas(root, cursor='cross', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        instruction = canvas.create_text(
            root.winfo_screenwidth() // 2, 50,
            text="Click and drag to select the maze region\nPress ESC to cancel",
            fill='white', font=('Arial', 16, 'bold')
        )
        
        def on_mouse_down(event):
            self.start_x = event.x
            self.start_y = event.y
            self.rect = canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline='red', width=3
            )
        
        def on_mouse_drag(event):
            if self.rect:
                canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
        
        def on_mouse_up(event):
            self.end_x = event.x
            self.end_y = event.y
            root.quit()
            root.destroy()
        
        def on_escape(event):
            self.start_x = None
            root.quit()
            root.destroy()
        
        canvas.bind('<ButtonPress-1>', on_mouse_down)
        canvas.bind('<B1-Motion>', on_mouse_drag)
        canvas.bind('<ButtonRelease-1>', on_mouse_up)
        root.bind('<Escape>', on_escape)
        
        root.mainloop()
        
        if self.start_x is None:
            return None
        
        # Normalize coordinates
        x1 = min(self.start_x, self.end_x)
        y1 = min(self.start_y, self.end_y)
        x2 = max(self.start_x, self.end_x)
        y2 = max(self.start_y, self.end_y)
        
        # Capture the selected region
        time.sleep(0.5)  # Brief delay for window to close
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        
        return screenshot


def capture_screen_region():
    """Allow user to select screen region to capture"""
    print("\n" + "="*60)
    print("SCREEN CAPTURE MODE")
    print("="*60)
    print("\nOptions:")
    print("1. Select specific region (click and drag)")
    print("2. Capture entire screen")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nA semi-transparent overlay will appear.")
        print("Click and drag to select the maze region.")
        print("Press ESC to cancel.\n")
        input("Press Enter to start selection...")
        
        selector = RegionSelector()
        screenshot = selector.select_region()
        
        if screenshot is None:
            print("Selection cancelled.")
            return None
        
        print("✓ Region selected successfully!")
        return screenshot
        
    elif choice == "2":
        print("\nCapturing entire screen in 3 seconds...")
        print("Make sure the maze is visible on screen!")
        time.sleep(3)
        screenshot = ImageGrab.grab()
        print("✓ Screen captured!")
        return screenshot
    else:
        print("Invalid choice.")
        return None


def main():
    print("=" * 60)
    print("AUTOMATED MAZE SOLVER WITH A* ALGORITHM")
    print("=" * 60)
    print("\nOptions:")
    print("1. Capture from screen")
    print("2. Load from image file")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    try:
        if choice == "1":
            # Capture from screen
            image = capture_screen_region()
            if image is None:
                print("\nScreen capture cancelled. Exiting.")
                return
        elif choice == "2":
            # Load from file
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select Maze Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
            )
            if not file_path:
                print("No file selected. Exiting.")
                return
            image = file_path
        else:
            print("Invalid choice. Exiting.")
            return
        
        # Detect maze
        print("\n[1/4] Detecting maze structure...")
        detector = MazeDetector(image)
        grid, start, goal = detector.detect_maze()
        
        print(f"✓ Maze detected: {len(grid[0])}x{len(grid)} cells")
        print(f"✓ Start position: {start}")
        print(f"✓ Goal position: {goal}")
        
        # Display maze
        print("\nMaze structure:")
        for row in grid:
            print(''.join(row))
        
        # Solve maze
        print("\n[2/4] Solving maze with A* algorithm...")
        solver = MazeSolver(grid, start, goal)
        actions = solver.solve()
        
        if not actions:
            print("✗ No solution found!")
            return
        
        print(f"✓ Solution found: {len(actions)} moves")
        print(f"✓ Path: {' → '.join(actions)}")
        
        # Ask for execution
        print("\n[3/4] Ready to execute solution")
        delay = float(input("Enter delay between moves in seconds (default 1.0): ") or "1.0")
        
        execute = input("\nExecute solution? (y/n): ").strip().lower()
        
        if execute == 'y':
            print("\n[4/4] Executing solution...")
            controller = MazeController(actions, delay)
            controller.execute()
        else:
            print("\nSolution not executed. Exiting.")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




# import cv2
# import numpy as np
# import pyautogui
# import time
# from heapq import heappush, heappop
# from PIL import Image, ImageGrab
# import tkinter as tk
# from tkinter import messagebox, filedialog

# # Disable pyautogui failsafe for smooth automation
# pyautogui.FAILSAFE = True
# pyautogui.PAUSE = 0.1


# class Node:
#     """Node for A* pathfinding"""
#     def __init__(self, state, parent, action=None, cost=0):
#         self.state = state
#         self.parent = parent
#         self.action = action
#         self.cost = cost


# class AStarFrontier:
#     """A* frontier using priority queue"""
#     def __init__(self, goal):
#         self.frontier = []
#         self.goal = goal
#         self.counter = 0

#     def heuristic(self, state):
#         """Manhattan distance heuristic"""
#         return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

#     def add(self, node):
#         f = node.cost + self.heuristic(node.state)
#         heappush(self.frontier, (f, self.counter, node))
#         self.counter += 1

#     def remove(self):
#         if self.empty():
#             raise Exception("Empty frontier")
#         return heappop(self.frontier)[2]

#     def empty(self):
#         return len(self.frontier) == 0

#     def contains_state(self, state):
#         for item in self.frontier:
#             node = item[2]
#             if node.state == state:
#                 return True
#         return False


# class MazeDetector:
#     """Detects and processes maze from image"""
    
#     def __init__(self, image):
#         if isinstance(image, str):
#             self.original = cv2.imread(image)
#         else:
#             self.original = np.array(image)
#             self.original = cv2.cvtColor(self.original, cv2.COLOR_RGB2BGR)
        
#         self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
#         self.grid = None
#         self.start = None
#         self.goal = None
#         self.cell_size = None
        
#     def detect_maze(self):
#         """Detect maze structure from image"""
#         # Apply threshold to get binary image
#         _, binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)
        
#         # Find contours to detect maze boundaries
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if not contours:
#             raise Exception("No maze detected in image")
        
#         # Get the largest contour (assumed to be the maze)
#         maze_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(maze_contour)
        
#         # Crop to maze area
#         maze_region = binary[y:y+h, x:x+w]
        
#         # Estimate cell size by finding grid lines
#         self.cell_size = self._estimate_cell_size(maze_region)
        
#         if self.cell_size < 5:
#             self.cell_size = 20  # Default fallback
        
#         # Create grid representation
#         self.grid = self._create_grid(maze_region)
        
#         # Detect start and goal
#         self._detect_start_goal()
        
#         return self.grid, self.start, self.goal
    
#     def _estimate_cell_size(self, image):
#         """Estimate cell size from grid lines"""
#         height, width = image.shape
        
#         # Sample horizontal lines
#         mid_row = height // 2
#         row_data = image[mid_row, :]
        
#         # Find transitions (black to white or vice versa)
#         transitions = []
#         for i in range(1, len(row_data)):
#             if row_data[i] != row_data[i-1]:
#                 transitions.append(i)
        
#         if len(transitions) > 1:
#             # Calculate average distance between transitions
#             distances = [transitions[i+1] - transitions[i] for i in range(len(transitions)-1)]
#             if distances:
#                 avg_cell_size = np.median(distances)
#                 return int(avg_cell_size)
        
#         return 20  # Default
    
#     def _create_grid(self, image):
#         """Convert image to grid representation"""
#         height, width = image.shape
        
#         grid_height = height // self.cell_size
#         grid_width = width // self.cell_size
        
#         grid = []
#         for row in range(grid_height):
#             grid_row = []
#             for col in range(grid_width):
#                 # Sample center of cell
#                 y = row * self.cell_size + self.cell_size // 2
#                 x = col * self.cell_size + self.cell_size // 2
                
#                 if y < height and x < width:
#                     pixel_value = image[y, x]
#                     # Black = wall, White = path
#                     if pixel_value < 128:
#                         grid_row.append('#')
#                     else:
#                         grid_row.append(' ')
#                 else:
#                     grid_row.append('#')
#             grid.append(grid_row)
        
#         return grid
    
#     def _detect_start_goal(self):
#         """Detect start (usually green) and goal (usually red) positions"""
#         # Convert to HSV for color detection
#         hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        
#         # Green range for start (adjust as needed)
#         lower_green = np.array([40, 50, 50])
#         upper_green = np.array([80, 255, 255])
#         green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
#         # Red range for goal (adjust as needed)
#         lower_red1 = np.array([0, 50, 50])
#         upper_red1 = np.array([10, 255, 255])
#         lower_red2 = np.array([170, 50, 50])
#         upper_red2 = np.array([180, 255, 255])
#         red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#         red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#         red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
#         # Find start position
#         green_points = cv2.findNonZero(green_mask)
#         if green_points is not None:
#             avg_point = np.mean(green_points, axis=0)[0]
#             grid_x = int(avg_point[0] // self.cell_size)
#             grid_y = int(avg_point[1] // self.cell_size)
#             self.start = (grid_x, grid_y)
#             if 0 <= grid_y < len(self.grid) and 0 <= grid_x < len(self.grid[0]):
#                 self.grid[grid_y][grid_x] = 'A'
        
#         # Find goal position
#         red_points = cv2.findNonZero(red_mask)
#         if red_points is not None:
#             avg_point = np.mean(red_points, axis=0)[0]
#             grid_x = int(avg_point[0] // self.cell_size)
#             grid_y = int(avg_point[1] // self.cell_size)
#             self.goal = (grid_x, grid_y)
#             if 0 <= grid_y < len(self.grid) and 0 <= grid_x < len(self.grid[0]):
#                 self.grid[grid_y][grid_x] = 'B'
        
#         # If start/goal not detected by color, find first/last empty cells
#         if self.start is None or self.goal is None:
#             self._find_start_goal_by_position()
    
#     def _find_start_goal_by_position(self):
#         """Find start and goal by position (top-left and bottom-right empty cells)"""
#         if self.start is None:
#             for y in range(len(self.grid)):
#                 for x in range(len(self.grid[0])):
#                     if self.grid[y][x] == ' ':
#                         self.start = (x, y)
#                         self.grid[y][x] = 'A'
#                         break
#                 if self.start:
#                     break
        
#         if self.goal is None:
#             for y in range(len(self.grid)-1, -1, -1):
#                 for x in range(len(self.grid[0])-1, -1, -1):
#                     if self.grid[y][x] == ' ':
#                         self.goal = (x, y)
#                         self.grid[y][x] = 'B'
#                         break
#                 if self.goal:
#                     break


# class MazeSolver:
#     """Solves maze using A* algorithm"""
    
#     def __init__(self, grid, start, goal):
#         self.grid = grid
#         self.start = start
#         self.goal = goal
#         self.height = len(grid)
#         self.width = len(grid[0]) if grid else 0
#         self.explored = set()
        
#     def neighbors(self, state):
#         """Get valid neighbors of current state"""
#         x, y = state
#         candidates = [
#             ("right", (x+1, y)),
#             ("left", (x-1, y)),
#             ("down", (x, y+1)),
#             ("up", (x, y-1))
#         ]
        
#         result = []
#         for action, (nx, ny) in candidates:
#             if 0 <= ny < self.height and 0 <= nx < len(self.grid[ny]):
#                 if self.grid[ny][nx] != '#':
#                     result.append((action, (nx, ny)))
#         return result
    
#     def solve(self):
#         """Solve maze using A* algorithm"""
#         frontier = AStarFrontier(self.goal)
#         start_node = Node(self.start, None, None, 0)
#         frontier.add(start_node)
        
#         while not frontier.empty():
#             node = frontier.remove()
            
#             if node.state == self.goal:
#                 return self._reconstruct_path(node)
            
#             self.explored.add(node.state)
            
#             for action, state in self.neighbors(node.state):
#                 if state not in self.explored and not frontier.contains_state(state):
#                     child = Node(
#                         state=state,
#                         parent=node,
#                         action=action,
#                         cost=node.cost + 1
#                     )
#                     frontier.add(child)
        
#         return []  # No solution found
    
#     def _reconstruct_path(self, node):
#         """Reconstruct path from goal to start"""
#         actions = []
#         while node.parent is not None:
#             actions.append(node.action)
#             node = node.parent
#         return actions[::-1]


# class MazeController:
#     """Controls maze navigation using keyboard"""
    
#     def __init__(self, actions, delay=1.0):
#         self.actions = actions
#         self.delay = delay
        
#     def execute(self):
#         """Execute the solution by pressing arrow keys"""
#         print(f"\nStarting maze navigation in 3 seconds...")
#         print(f"Total moves: {len(self.actions)}")
#         print("Move your mouse to top-left corner to emergency stop!")
        
#         time.sleep(3)
        
#         for i, action in enumerate(self.actions):
#             print(f"Move {i+1}/{len(self.actions)}: {action}")
            
#             if action == "up":
#                 pyautogui.press('up')
#             elif action == "down":
#                 pyautogui.press('down')
#             elif action == "left":
#                 pyautogui.press('left')
#             elif action == "right":
#                 pyautogui.press('right')
            
#             time.sleep(self.delay)
        
#         print("\n✓ Maze solved successfully!")


# def capture_screen_region():
#     """Allow user to select screen region to capture"""
#     print("Select the maze region on your screen...")
#     root = tk.Tk()
#     root.withdraw()
    
#     # Get screen dimensions
#     screen_width, screen_height = pyautogui.size()
    
#     messagebox.showinfo("Screen Capture", 
#                        "Click OK, then you have 3 seconds to focus on the maze window.\n"
#                        "The entire screen will be captured.")
    
#     time.sleep(3)
#     screenshot = ImageGrab.grab()
    
#     return screenshot


# def main():
#     print("=" * 60)
#     print("AUTOMATED MAZE SOLVER WITH A* ALGORITHM")
#     print("=" * 60)
#     print("\nOptions:")
#     print("1. Capture from screen")
#     print("2. Load from image file")
    
#     choice = input("\nEnter choice (1 or 2): ").strip()
    
#     try:
#         if choice == "1":
#             # Capture from screen
#             image = capture_screen_region()
#         elif choice == "2":
#             # Load from file
#             root = tk.Tk()
#             root.withdraw()
#             file_path = filedialog.askopenfilename(
#                 title="Select Maze Image",
#                 filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
#             )
#             if not file_path:
#                 print("No file selected. Exiting.")
#                 return
#             image = file_path
#         else:
#             print("Invalid choice. Exiting.")
#             return
        
#         # Detect maze
#         print("\n[1/4] Detecting maze structure...")
#         detector = MazeDetector(image)
#         grid, start, goal = detector.detect_maze()
        
#         print(f"✓ Maze detected: {len(grid[0])}x{len(grid)} cells")
#         print(f"✓ Start position: {start}")
#         print(f"✓ Goal position: {goal}")
        
#         # Display maze
#         print("\nMaze structure:")
#         for row in grid:
#             print(''.join(row))
        
#         # Solve maze
#         print("\n[2/4] Solving maze with A* algorithm...")
#         solver = MazeSolver(grid, start, goal)
#         actions = solver.solve()
        
#         if not actions:
#             print("✗ No solution found!")
#             return
        
#         print(f"✓ Solution found: {len(actions)} moves")
#         print(f"✓ Path: {' → '.join(actions)}")
        
#         # Ask for execution
#         print("\n[3/4] Ready to execute solution")
#         delay = float(input("Enter delay between moves in seconds (default 1.0): ") or "1.0")
        
#         execute = input("\nExecute solution? (y/n): ").strip().lower()
        
#         if execute == 'y':
#             print("\n[4/4] Executing solution...")
#             controller = MazeController(actions, delay)
#             controller.execute()
#         else:
#             print("\nSolution not executed. Exiting.")
    
#     except Exception as e:
#         print(f"\n✗ Error: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()