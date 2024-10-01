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


class Entity:

    def __init__(self, mesh: Mesh, material: Material,
                 positions: list[float],
                 eulers: list[float],
                 velocities: list[float],
                 size: list[float]):
        self.positions = np.array(positions, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)

        self.mesh = mesh
        self.material = material

        self.active = True

    def update(self):
        if self.active:
            self.positions += self.velocities

            self.transform_mat = pyrr.matrix44.create_identity(dtype=np.float32)

            self.transform_mat = pyrr.matrix44.multiply(
                self.transform_mat,
                pyrr.matrix44.create_from_scale(self.size, dtype=np.float32)
            )

            self.transform_mat = pyrr.matrix44.multiply(
                self.transform_mat,
                pyrr.matrix44.create_from_eulers(eulers=np.radians(self.eulers), dtype=np.float32)
            )
            self.transform_mat = pyrr.matrix44.multiply(
                self.transform_mat,
                pyrr.matrix44.create_from_translation(self.positions, dtype=np.float32)
            )

    def draw(self, shader, model_mat_location):
        if self.active:
            glUseProgram(shader)
            self.material.use()

            glUniformMatrix4fv(model_mat_location, 1, GL_FALSE, self.transform_mat)
            glBindVertexArray(self.mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.mesh.vertex_count)
        else:
            pass

class Bullet(Entity):

    def __init__(self, mesh: Mesh, material: Material,
             positions: list[float],
             eulers: list[float],
             velocities: list[float],
             size: list[float]):
        super().__init__(mesh=mesh, material=material, positions=positions, eulers=eulers, velocities=velocities, size=size)

    def update(self, entities: dict=dict()):
        super().update()
        if self.active:
            for key, entity in entities.items():
                if (entity.positions[0] < self.positions[0] < entity.positions[0] + entity.size[0] and
                    entity.positions[1] < self.positions[1] < entity.positions[1] + entity.size[1] and
                    entity.positions[2] < self.positions[2] < entity.positions[2] + entity.size[2]):
                    entity.active = False


            if self.velocities[0] > 0:
                self.eulers[2] = 90
            if self.velocities[1] > 0:
                self.eulers[0] = 90
            if self.velocities[1] > 0:
                self.eulers[1] = 90


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

        self.cube_mesh = Mesh("models/cube.obj")
        self.bullet_mesh = Mesh("models/test_bullet.obj")

        self.wood_texture = Material("textures/wood.jpeg")
        self.cat_texture = Material("textures/cat.png")

        self.entitys = {
            "cube1": Entity(self.cube_mesh, self.wood_texture, [0.0, 0.0, -5.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [1, 1, 1]),
            "cube2": Entity(self.cube_mesh, self.cat_texture, [2.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 1, 0.2])
        }

        self.bullets = {
            "bullet1": Bullet(self.bullet_mesh, self.wood_texture, [0.0, 0.0, -3.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.0], [0.2, 0.2, 0.2]),
        }

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

            pressed_keys = pg.key.get_pressed()

            if pressed_keys[pg.K_w]:
                self.entitys["cube1"].positions[2] -= 0.04
            if pressed_keys[pg.K_s]:
                self.entitys["cube1"].positions[2] += 0.04
            if pressed_keys[pg.K_a]:
                self.entitys["cube1"].positions[0] -= 0.04
            if pressed_keys[pg.K_d]:
                self.entitys["cube1"].positions[0] += 0.04
            if pressed_keys[pg.K_SPACE]:
                self.entitys["cube1"].positions[1] += 0.04
            if pressed_keys[pg.K_LSHIFT]:
                self.entitys["cube1"].positions[1] -= 0.04


            # self.entitys["cube1"].eulers[0] += 0.2
            # if self.entitys["cube1"].eulers[0] > 360:
            #     self.entitys["cube1"].eulers[0] -= 360


            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            for entity in self.entitys.values():
                entity.update()
                entity.draw(self.shader, self.model_matrix_location)

            for bullet in self.bullets.values():
                bullet.update(self.entitys)
                bullet.draw(self.shader, self.model_matrix_location)

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
