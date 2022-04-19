import glfw
from OpenGL.GL import *
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import json
from pixel_paint_model import *
from pixel_paint_controller import Controller

#grid_size = int(sys.argv[1])
#palette = open(str(sys.argv[2]))
#palette_data = json.load(palette)

def window_resize(window, width, height):
  glViewport(0, 0, width, height)
  #projection = tr.perspective(45, width/height, 0.01, 100)
  #glUniformMatrix4fv(transform_loc, 1, GL_FALSE, projection)
 
def cursorPositionCallback(window, x_pos, y_pos):
  print(x_pos, y_pos)

# Initialize glfw
if not glfw.init():
  raise Exception("glfw can not be initialized!")

# Create window
window = glfw.create_window(600, 600, "Pixel Paint", None, None)

# Check if window was created
if not window:
  glfw.terminate()
  raise Exception("glfw window can not be created!")

# Set window's position
glfw.set_window_pos(window, 600, 150)

# Resize callback
glfw.set_window_size_callback(window, window_resize)

glfw.set_cursor_pos_callback(window, cursorPositionCallback)

# Make the context current
glfw.make_context_current(window)


controller = Controller()

pipeline = es.SimpleTransformShaderProgram()
shader = pipeline.shaderProgram
glUseProgram(shader)



#model_loc = glGetUniformLocation(shader, "model")
#proj_loc = glGetUniformLocation(shader, "projection")
#view_loc = glGetUniformLocation(shader, "view")
transform_loc = glGetUniformLocation(shader, "transform")

glClearColor(0.5, 0.5, 0.5, 0.1) # Window colors

# Connecting the callback function 'on_key' to handle keyboard events
glfw.set_key_callback(window, controller.on_key)

# Make objects
W = 16 # Cantidad de columnas 
H = 16 # Cantidad de filas
imgData = np.zeros((W, H, 3), dtype=np.uint8)
imageSize = imgData.shape

palette = [[1, 0, 0] , [0, 1, 0], [0, 0, 1]]
grid = Grid(imageSize, pipeline)
print(grid)
palette = Palette(pipeline)
imgData[:, :] = np.array([1, 1, 1], dtype=np.uint8)
grid.setMatrix(imgData)

# Control objects
controller.set_grid(grid)
controller.set_palette(palette)

grid.change_color()

# The main loop
while not glfw.window_should_close(window):
  glfw.poll_events()
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) # Change window color

  palette.draw(pipeline)
  grid.draw(pipeline)

  glfw.swap_buffers(window)
  
# Terminate glfw
glfw.terminate()