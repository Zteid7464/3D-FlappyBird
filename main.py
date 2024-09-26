import pygame as pg
import numpy as np
from OpenGL.GL import *
import ctypes
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import math


class Material:

    def __init__(self, filepath: str):
        # generate one texture
        self.texture = glGenTextures(1)
        # bind the texture
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # set rapping mode for s t coordinates
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # set minifying and magnifying filters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # load the actual image
        image = pg.image.load(filepath).convert_alpha()
        image_width, image_height = image.get_rect().size
        image_data = pg.image.tostring(image, "RGBA")
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))


class Mesh:

    def __init__(self, filename):
        # x, y, z, s, t
        self.vertices = self.load_mesh(filename)

        self.vertex_count = len(self.vertices) // 8

        self.vertices = np.array(self.vertices, dtype=np.float32)


        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # vertices
        self.vbo = glGenVertexArrays(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        # texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

    def read_vertice_data(self, words: list[str]) -> list[float]:
        return [float(words[i + 1]) for i in range(len(words) - 1)]

    def make_corner(self, corner_description: str, v: list[list[float]], vt: list[list[float]], vn: list[list[float]], vertices: list[float]) -> None:
        v_vt_vn = corner_description.split("/")
        for element in v[int(v_vt_vn[0]) - 1]:
            vertices.append(element)
        for element in vt[int(v_vt_vn[1]) - 1]:
            vertices.append(element)
        for element in vn[int(v_vt_vn[2]) - 1]:
            vertices.append(element)

    def read_face_data(self, words: list[str], v: list[list[float]], vt: list[list[float]], vn: list[list[float]], vertices: list[float]) -> None:
        triangle_count = len(words) - 3

        for i in range(triangle_count):
            self.make_corner(words[1], v, vt, vn, vertices)
            self.make_corner(words[2 + i], v, vt, vn, vertices)
            self.make_corner(words[3 + i], v, vt, vn, vertices)


    def load_mesh(self, filename: str) -> list[float]:
        v = []
        vt = []
        vn = []

        vertices = []

        with open(filename, "r") as obj_file:
            line = obj_file.readline()

            while line:
                words = line.split(" ")
                if words[0] == "v":
                   v.append(self.read_vertice_data(words))
                elif words[0] == "vt":
                   vt.append(self.read_vertice_data(words))
                elif words[0] == "vn":
                   vn.append(self.read_vertice_data(words))
                elif words[0] == "f":
                    self.read_face_data(words, v, vt, vn, vertices)
                line = obj_file.readline()

        return vertices

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


class Model:

    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)


class App:

    def __init__(self):
        # initialize pygame
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        pg.display.set_caption("OpenGl python test test")
        self.clock = pg.time.Clock()

        # initialize opengl
        glClearColor(0.1, 0.2, 0.2, 1)
        # enable alpha transparency
        glEnable(GL_BLEND)
        # enable depth test
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.shader = self.create_shader("shaders/vertex.glsl", "shaders/fragment.glsl")
        glUseProgram(self.shader)
        # get the sampler to sample texture 0
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)

        self.models = {
            "cube1": Model(
                position=[0, 0, -5],
                eulers=[0, 0, 0]),
            "cube2": Model(
                position=[2, 0, -5],
                eulers=[0, 0, 0]
            )
        }

        self.cube_mesh = Mesh("models/cube.obj")

        self.wood_texture = Material("textures/wood.jpeg")

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45, aspect=640/480,
            near=0.1, far=10, dtype=np.float32
        )

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )

        self.model_matrix_location = glGetUniformLocation(self.shader, "model")

        self.main_loop()

    def create_shader(self, vertex_filepath: str, fragment_filepath: str):
        with open(vertex_filepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragment_filepath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

        return shader

    def main_loop(self):
        running = True

        while running:
            # check for events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # update cube
            # self.cube.eulers[2] += 0.2
            # self.cube.eulers[1] += 0.2
            self.models["cube1"].eulers[0] += 0.2
            self.models["cube2"].position[0] = math.sin(self.models["cube1"].eulers[0]) / 2
            if self.models["cube1"].eulers[2] > 360:
                # self.cube.eulers[2] -= 360
                # self.cube.eulers[1] -= 360
                self.models["cube1"].eulers[0] -= 360

            glUseProgram(self.shader)
            self.wood_texture.use()

            for model in self.models.values():

                model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
                model_transform = pyrr.matrix44.multiply(
                    m1=model_transform,
                    m2=pyrr.matrix44.create_from_eulers(
                        eulers=np.radians(model.eulers),
                        dtype=np.float32
                    )
                )

                model_transform = pyrr.matrix44.multiply(
                    m1=model_transform,
                    m2=pyrr.matrix44.create_from_translation(
                        vec=model.position,
                        dtype=np.float32
                    )
                )

                glUniformMatrix4fv(self.model_matrix_location, 1, GL_FALSE, model_transform)
                glBindVertexArray(self.cube_mesh.vao)
                glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)

            # display the fps
            pg.display.set_caption(str(self.clock.get_fps()))

            pg.display.flip()

            # setting the FPS
            self.clock.tick(60)
        self.quit()

    def quit(self):

        self.cube_mesh.destroy()
        self.wood_texture.destroy()
        glDeleteProgram(self.shader)
        pg.quit()


if __name__ == "__main__":
    app = App()
