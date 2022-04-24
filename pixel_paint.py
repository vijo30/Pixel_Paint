import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders
import glfw
from PIL import Image, ImageOps
import sys
import json



k = int(sys.argv[1])
nombre_archivo_json = sys.argv[2]
file_name = sys.argv[3]


print(f'El nombre del archivo json es: {nombre_archivo_json}')

with open(nombre_archivo_json) as json_file:
    data = json.load(json_file)
    # Queda como un diccionario

    print(f'El transparente es {data["transparent"]}')
    print(f'La paleta de colores es {data["palette"]}')

    for index, color in enumerate(data["palette"]):
        print(f'El color {index+1} de la paleta es: {color}')

transparent = data["transparent"]
colors = data["palette"]



# A simple class container to store vertices and indices that define a shape
class Shape:
    def __init__(self, vertices, indices, textureFileName=None):
        self.vertices = vertices
        self.indices = indices
        self.textureFileName = textureFileName


# We will use 32 bits data, so we have 4 bytes
# 1 byte = 8 bits
SIZE_IN_BYTES = 4


# A simple class container to reference a shape on GPU memory
class GPUShape:
    def __init__(self):
        self.vao = 0
        self.vbo = 0
        self.ebo = 0
        self.texture = 0
        self.size = 0



class SimpleShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330

            in vec3 position;
            in vec3 color;

            out vec3 newColor;
            void main()
            {
                gl_Position = vec4(position, 1.0f);
                newColor = color;
            }
            """

        fragment_shader = """
            #version 330
            in vec3 newColor;

            out vec4 outColor;
            void main()
            {
                outColor = vec4(newColor, 1.0f);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    def drawShape(self, shape, mode=GL_TRIANGLES):
        assert isinstance(shape, GPUShape)

        # Binding the proper buffers
        glBindVertexArray(shape.vao)
        glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)

        # 3d vertices + rgb color specification => 3*4 + 3*4 = 24 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        # Render the active element buffer with the active shader program
        glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)


class SimpleTextureShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330

            in vec3 position;
            in vec2 texCoords;

            out vec2 outTexCoords;

            void main()
            {
                gl_Position = vec4(position, 1.0f);
                outTexCoords = texCoords;
            }
            """

        fragment_shader = """
            #version 330

            in vec2 outTexCoords;

            out vec4 outColor;

            uniform sampler2D samplerTex;

            void main()
            {
                outColor = texture(samplerTex, outTexCoords);
            }
            """

        # Binding artificial vertex array object for validation
        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    def drawShape(self, shape, mode=GL_TRIANGLES):
        assert isinstance(shape, GPUShape)

        # Binding the proper buffers
        glBindVertexArray(shape.vao)
        glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)
        glBindTexture(GL_TEXTURE_2D, shape.texture)

        # 3d vertices + 2d texture coordinates => 3*4 + 2*4 = 20 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        texCoords = glGetAttribLocation(self.shaderProgram, "texCoords")
        glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(texCoords)

        # Render the active element buffer with the active shader program
        glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)


def toGPUShape(shape):
    assert isinstance(shape, Shape)

    vertexData = np.array(shape.vertices, dtype=np.float32)
    indices = np.array(shape.indices, dtype=np.uint32)

    # Here the new shape will be stored
    gpuShape = GPUShape()

    gpuShape.size = len(shape.indices)
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape


# A class to store the application control
class Controller:

    def __init__(self):
        self.fillPolygon = True
        self.showGrid = True
        self.leftClickOn = False
        self.rightClickOn = False
        self.mousePos = (0.0, 0.0)

controller = Controller()

def createGPUTextureQuad():
    vertices = [
        #   positions   texture
        -1, -1, 0, 1, 0,
        1, -1, 0, 1, 1,
        1, 1, 0, 0, 1,
        -1, 1, 0, 0, 0]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 2,
        2, 3, 0]

    return Shape(vertices, indices)


def createGrid(Nx, Ny):
    vertices = []
    indices = []
    index = 0

    # cols
    for x in np.linspace(-1, 1, Nx + 1, True):
        vertices += [x, -1, 0] + [0, 0, 0]
        vertices += [x, 1, 0] + [0, 0, 0]
        indices += [index, index + 1]
        index += 2

    # rows
    for y in np.linspace(-1, 1, Ny + 1, True):
        vertices += [-1, y, 0] + [0, 0, 0]
        vertices += [1, y, 0] + [0, 0, 0]
        indices += [index, index + 1]
        index += 2

    return Shape(vertices, indices)


def save_image(imgData):
  img = Image.fromarray(imgData) 
  img = img.rotate(-90)
  img = ImageOps.mirror(img)
  img.save(file_name)
  img.show()



def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return

    if key == glfw.KEY_SPACE:
        controller.showGrid = not controller.showGrid

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
        
    if key == glfw.KEY_S:
      save_image(imgData[0:k, 0:k, :])
        





def cursor_pos_callback(window, x, y):
    global controller
    controller.mousePos = (x, y)


def mouse_button_callback(window, button, action, mods):
    global controller

    """
    glfw.MOUSE_BUTTON_1: left click
    glfw.MOUSE_BUTTON_2: right click
    glfw.MOUSE_BUTTON_3: scroll click
    """
    if (action == glfw.PRESS):
        if (button == glfw.MOUSE_BUTTON_1):
            controller.leftClickOn = True
            print("Mouse click - button 1")
            print("Color:", imgData[int(gridPosX) : int(gridPosX) + 1 , int(gridPosY) : int(gridPosY) + 1, :][0][0])


            

        if (button == glfw.MOUSE_BUTTON_2):
            controller.rightClickOn = True
            print("Mouse click - button 2:", int(gridPosX), int(gridPosY))

        if (button == glfw.MOUSE_BUTTON_3):
            print("Mouse click - button 3")

    elif (action == glfw.RELEASE):
        if (button == glfw.MOUSE_BUTTON_1):
            controller.leftClickOn = False
            
        if (button == glfw.MOUSE_BUTTON_2):
          controller.rightClickOn = False


#def scroll_callback(window, x, y):
    #print("Mouse scroll:", x, y)








win_height = 800
win_width = 800
# Initialize glfw
if not glfw.init():
  raise Exception("glfw can not be initialized!")

# Prevent resizing
glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)


window = glfw.create_window(win_width, win_height, "Pixel Paint", None, None)

# Check if window was created
if not window:
  glfw.terminate()
  raise Exception("glfw window can not be created!")


def n_adjustment(k):
    if k >= 10:
        n = k + 1
        return n
    if k < 10:
        n = 11
        return n




# Cantidad de pixeles
#k = 16
n = n_adjustment(k)
adjustment = win_width / n


imgData = np.zeros((n, n, 4), dtype=np.uint8)
imgData[:, :, :] = np.array([88, 88, 88, 255])
imgData[0:k, 0:k, :] = np.array([transparent[0], transparent[1], transparent[2], 0], dtype=np.uint8)


def addColorPalette(palette):
    for i in range(0, len(palette)):
        imgData[n-1:n, 0:1, :] = np.array([transparent[0], transparent[1], transparent[2], 0])
        imgData[n-1:n, i+1:i+2, :] = np.array([palette[i][0], palette[i][1], palette[i][2], 255])

    



addColorPalette(colors)


imageSize = (n, n)


glfw.make_context_current(window)
glfw.set_key_callback(window, on_key)
glfw.set_cursor_pos_callback(window, cursor_pos_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)
#glfw.set_scroll_callback(window, scroll_callback)




pipeline = SimpleTextureShaderProgram()

colorPipeline = SimpleShaderProgram()


gpuShape = toGPUShape(createGPUTextureQuad())
gpuGrid = toGPUShape(createGrid(imageSize[0], imageSize[1]))

gpuShape.texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, gpuShape.texture)

# texture wrapping params
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

# texture filtering params
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

internalFormat = GL_RGBA
format = GL_RGBA

glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, imageSize[1], imageSize[0], 0, format,
              GL_UNSIGNED_BYTE, imgData)



r = 255
g = 0
b = 0
a = 255


while not glfw.window_should_close(window):
    glfw.poll_events()
    
    # Getting the mouse location relative to the grid
    gridPosX = controller.mousePos[0] / adjustment
    gridPosY = controller.mousePos[1] / adjustment

    
    if controller.leftClickOn and int(gridPosX) > n-2 and int(gridPosY) < len(colors)+1:
      r = imgData[int(gridPosX) : int(gridPosX) + 1 , int(gridPosY) : int(gridPosY) + 1, :][0][0][0]
      g = imgData[int(gridPosX) : int(gridPosX) + 1 , int(gridPosY) : int(gridPosY) + 1, :][0][0][1]
      b = imgData[int(gridPosX) : int(gridPosX) + 1 , int(gridPosY) : int(gridPosY) + 1, :][0][0][2]
      a = imgData[int(gridPosX) : int(gridPosX) + 1 , int(gridPosY) : int(gridPosY) + 1, :][0][0][3]

    
    if controller.leftClickOn and (int(gridPosX) < k and int(gridPosY) < k):
      imgData[int(gridPosX) : int(gridPosX) + 1 , int(gridPosY) : int(gridPosY) + 1, :] = np.array([r, g, b, a], dtype=np.uint8)
      glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, imageSize[1], imageSize[0], 0, format,
              GL_UNSIGNED_BYTE, imgData)
      

    
    
    if controller.rightClickOn  and (int(gridPosX) < k and int(gridPosY) < k):
      imgData[int(gridPosX) : int(gridPosX) + 1 , int(gridPosY) : int(gridPosY) + 1, :] = np.array([transparent[0], transparent[1], transparent[2], 0], dtype=np.uint8)
      glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, imageSize[1], imageSize[0], 0, format,
              GL_UNSIGNED_BYTE, imgData)
      
    
    if controller.fillPolygon:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(pipeline.shaderProgram)
    pipeline.drawShape(gpuShape)

    if controller.showGrid:
        glUseProgram(colorPipeline.shaderProgram)
        colorPipeline.drawShape(gpuGrid, GL_LINES)

    
    # Once the render is done, buffers are swapped, showing only the complete scene.
    glfw.swap_buffers(window)


glfw.terminate()
