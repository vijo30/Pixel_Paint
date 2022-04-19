import numpy as np
import grafica.basic_shapes as bs
import grafica.easy_shaders as es 
import grafica.gpu_shape as gs
import grafica.transformations as tr 
import grafica.scene_graph as sg
from OpenGL.GL import *
import OpenGL.GL.shaders
import glfw







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
        self.drawPalette = True


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


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return

    if key == glfw.KEY_SPACE:
        controller.showGrid = not controller.showGrid

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)



def setMatrix(matrix):
    assert (imageSize[0] == matrix.shape[0])
    assert (imageSize[1] == matrix.shape[1])

    # RGB 8 bits for each channel 
    assert (matrix.shape[2] == 3)
    assert (matrix.dtype == np.uint8)

    imgData = matrix.reshape((matrix.shape[0] * matrix.shape[1], 3))
    
def create_gpu(shape, pipeline):
    gpu = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpu)
    gpu.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpu


class Palette:
  
  
  def __init__(self, pipeline):
    gpu_canvas_quad = create_gpu(bs.createColorQuad(1.0, 1.0, 1.0), pipeline)
    gpu_color_quad = create_gpu(bs.createColorQuad(0.5, 0.5, 0.5), pipeline)
    gpu_color2_quad = create_gpu(bs.createColorQuad(1.0, 0.0, 0.0), pipeline)
 
    
    canvas = sg.SceneGraphNode('canvas')
    canvas.transform = tr.scale(0.3, 2, 1)
    canvas.childs += [gpu_canvas_quad]
    
    color = sg.SceneGraphNode('color')
    color.transform = tr.matmul([tr.translate(0, 0.8, 0), tr.scale(0.2, 0.1, 1)])
    color.childs += [gpu_color_quad]

    color2 = sg.SceneGraphNode('color2')
    color2.transform = tr.matmul([tr.translate(0, 0.65, 0), tr.scale(0.2, 0.1, 1)])
    color2.childs += [gpu_color2_quad]
    
    
    
    palette = sg.SceneGraphNode('palette')
    palette.transform = tr.translate(0.85, 0, 0)
    palette.childs += [canvas, color, color2]
    
    transform_palette = sg.SceneGraphNode('paletteTR')
    transform_palette.transform = tr.identity()
    transform_palette.childs += [palette]
    
    self.model = transform_palette
    
  def draw(self, pipeline):
    sg.drawSceneGraphNode(self.model, pipeline, 'transform')



# Initialize glfw
if not glfw.init():
  raise Exception("glfw can not be initialized!")


window = glfw.create_window(600, 600, "Pixel Paint", None, None)

# Check if window was created
if not window:
  glfw.terminate()
  raise Exception("glfw window can not be created!")




glfw.make_context_current(window)
glfw.set_key_callback(window, on_key)




W = 20
H = 20
imgData = np.zeros((W, H, 3), dtype=np.uint8)
imgData[0:16, 0:16, :] = np.array([125, 125, 125], dtype=np.uint8)
imgData[0:4, 0:4, :] = np.array([0, 0, 255], dtype=np.uint8)
imgData[0:12, 4:8, :] = np.array([255, 0, 0], dtype=np.uint8)
imgData[4:12, 0:4, :] = np.array([255, 255, 255], dtype=np.uint8)

imageSize = (W, H)


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

internalFormat = GL_RGB
format = GL_RGB

glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, imageSize[1], imageSize[0], 0, format,
              GL_UNSIGNED_BYTE, imgData)



controller = Controller()



while not glfw.window_should_close(window):
    glfw.poll_events()
    

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
