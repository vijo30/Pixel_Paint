import glfw
from OpenGL.GL import *
import numpy as np
from math import sin, cos

# initialize glfw
if not glfw.init():
  raise Exception("glfw can not be initialized!")

# create window
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

# check if window was created
if not window:
  glfw.terminate()
  raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# make the context current
glfw.make_context_current(window)

glClearColor(0, 0.1, 0.1, 1) # window colors

# vertices and color of the triangle

vertices = [-0.5, -0.5, 0.0,
            0.5, -0.5, 0.0,
            0.0, 0.5, 0.0]

colors = [1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0]

# convert list to array

vertices = np.array(vertices, dtype=np.float32)
colors = np.array(colors, dtype=np.float32)

# define the array
glEnableClientState(GL_VERTEX_ARRAY)
glVertexPointer(3, GL_FLOAT, 0, vertices)

# colour the triangle
glEnableClientState(GL_COLOR_ARRAY)
glColorPointer(3, GL_FLOAT, 0, colors)

# the main loop
while not glfw.window_should_close(window):
  glfw.poll_events()
  
  glClear(GL_COLOR_BUFFER_BIT) # change window color
  
  ct = glfw.get_time() # returns the elapsed time, since init was called
  
  glLoadIdentity()
  glScale(abs(sin(ct)), abs(sin(ct)), 1)
  glRotate(sin(ct) * 45, 0, 0, 1)
  glTranslatef(sin(ct), cos(ct), 0)
  
  glDrawArrays(GL_TRIANGLES, 0, 3) # draw
  
  glfw.swap_buffers(window)
  
# terminate glfw
glfw.terminate()