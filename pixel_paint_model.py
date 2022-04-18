import numpy as np
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.gpu_shape as gs
from OpenGL.GL import *
from typing import List
import glfw

# A simple class container to store vertices and indices that define a shape
class Shape:
    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices

    def __str__(self):
        return "vertices: " + str(self.vertices) + "\n" \
                                                   "indices: " + str(self.indices)


class SceneGraphNodeForLines:
    """
    A simple class to handle a scene graph
    Each node represents a group of objects
    Each leaf represents a basic figure (GPUShape)
    To identify each node properly, it MUST have a unique name
    """

    def __init__(self, name):
        self.name = name
        self.transform = tr.identity()
        self.childs = []

    def clear(self):
        """Freeing GPU memory"""

        for child in self.childs:
            child.clear()
            
def drawSceneGraphNodeForLines(node, pipeline, transformName, parentTransform=tr.identity(), mode=GL_LINES):
    assert (isinstance(node, SceneGraphNodeForLines))

    # Composing the transformations through this path
    newTransform = np.matmul(parentTransform, node.transform)

    # If the child node is a leaf, it should be a GPUShape.
    # Hence, it can be drawn with drawCall
    if len(node.childs) == 1 and isinstance(node.childs[0], gs.GPUShape):
        leaf = node.childs[0]
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, transformName), 1, GL_TRUE, newTransform)
        pipeline.drawCall(leaf,mode)

    # If the child node is not a leaf, it MUST be a SceneGraphNode,
    # so this draw function is called recursively
    else:
        for child in node.childs:
            drawSceneGraphNodeForLines(child, pipeline, transformName, newTransform, mode)

def createGPUTextureQuad():
    vertices = [
    #   positions   texture
        -1, -1, 0,  1, 0,
         1, -1, 0,  1, 1,
         1,  1, 0,  0, 1,
        -1,  1, 0,  0, 0]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2,
         2, 3, 0]

    return Shape(vertices, indices)

    
    
def create_gpu(shape, pipeline):
    gpu = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpu)
    gpu.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpu


def createGrid(Nx, Ny):
  
    vertices = []
    indices = []
    index = 0

    # cols
    for x in np.linspace(-1, 0.7, Nx + 1, True):
        vertices += [x, -1, 0] + [0,0,0]
        vertices += [x,  0.7, 0] + [0,0,0]
        indices += [index, index+1]
        index += 2

    # rows
    for y in np.linspace(-1, 0.7, Ny + 1, True):
        vertices += [-1, y, 0] + [0,0,0]
        vertices += [ 0.7, y, 0] + [0,0,0]
        indices += [index, index+1]
        index += 2

    return Shape(vertices, indices)

def mouse_pos(window):
  x = glfw.get_cursor_pos(window)[0]/600
  y = glfw.get_cursor_pos(window)[1]/600
  return (x, y)


class Grid:
  

  def __init__(self, imageSize, pipeline):
      self.imageSize = imageSize
      # createGrid(self.imageSize[0], self.imageSize[1])  # Vertices and indices of the grid (numpy)
      self.vertex_data = createGrid(self.imageSize[0], self.imageSize[1]).vertices
      self.indices_data = createGrid(self.imageSize[0], self.imageSize[1]).indices
      gpuGrid = create_gpu(createGrid(self.imageSize[0], self.imageSize[1]), pipeline) # imageSize: size of numpy matrix
      grid = SceneGraphNodeForLines('grid')
      grid.transform = tr.identity()
      grid.childs += [gpuGrid]
      
      
      self.model = grid
       
  def setMatrix(self, matrix):
      assert(self.imageSize[0] == matrix.shape[0])
      assert(self.imageSize[1] == matrix.shape[1])
      
      # RGB 8 bits for each channel 
      assert(matrix.shape[2] == 3)
      assert(matrix.dtype == np.uint8)

      self.imgData = matrix.reshape((matrix.shape[0] * matrix.shape[1], 3))
  
  def draw(self, pipeline):
      drawSceneGraphNodeForLines(self.model, pipeline, 'transform')
    
  def draw_quad(self, r, g, b, pipeline, x, y, n):
    gpu_quad = create_gpu(bs.createColorQuad(r, g, b), pipeline)
    
    quads = sg.SceneGraphNode('quads')
    quads.transform = tr.matmul([tr.translate(x, y, 0), tr.scale(2/n+1, 2/n+1, 1)])
    quads.childs += quads.childs.append(gpu_quad)
    sg.drawSceneGraphNode(quads, pipeline, 'transform')
    


  
  def modifyModel(self):
    pass
      
  def update(self, color):
    pass
    
  def change_color(self):
      self.pipeline = es.SimpleTransformShaderProgram()
      self.gpuShape = create_gpu(bs.createColorQuad(0,1,0), self.pipeline)
      
  def change_actual_color(self):
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glUseProgram(self.pipeline.shaderProgram)
    self.pipeline.drawCall(self.gpuShape)

      
  
  def pick_color(self):
    pass
    
  


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



      
  