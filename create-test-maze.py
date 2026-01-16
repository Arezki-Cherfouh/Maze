from PIL import Image, ImageDraw

def create_test_maze():
    """Create a simple test maze image"""
    cell_size = 40
    maze = [
        "###############",
        "#A            #",
        "# ########### #",
        "#           # #",
        "########### # #",
        "#           # #",
        "# ########### #",
        "#             #",
        "# ########### #",
        "#           # #",
        "########### # #",
        "#            B#",
        "###############"
    ]
    
    width = len(maze[0]) * cell_size
    height = len(maze) * cell_size
    
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            x1 = x * cell_size
            y1 = y * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            if cell == '#':
                # Wall - black
                draw.rectangle([x1, y1, x2, y2], fill="black")
            elif cell == 'A':
                # Start - green
                draw.rectangle([x1, y1, x2, y2], fill="green")
            elif cell == 'B':
                # Goal - red
                draw.rectangle([x1, y1, x2, y2], fill="red")
            else:
                # Path - white
                draw.rectangle([x1, y1, x2, y2], fill="white")
    
    img.save("test_maze.png")
    print("Test maze created: test_maze.png")
    print(f"Maze size: {width}x{height} pixels")
    print("\nYou can use this image to test the automated maze solver!")

if __name__ == "__main__":
    create_test_maze()