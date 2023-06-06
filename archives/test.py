import pybullet as pb
import pybullet_data
import time
import os, glob, random

physicsClient = pb.connect(pb.GUI)
pb_data_path = pybullet_data.getDataPath()

pb.setAdditionalSearchPath(pb_data_path)
planeId = pb.loadURDF("plane.urdf")

soccer_ball = pb.loadURDF("soccerball.urdf",
                [0, 3, 5],
                pb.getQuaternionFromEuler([0, 0, 0]))

print("soccer_ball:", soccer_ball)

visualShapeId = pb.createVisualShape(
    shapeType=pb.GEOM_MESH,
    fileName=os.path.join(pb_data_path, 'random_urdfs/000/000.obj'),
    rgbaColor=None,
    meshScale=[0.1, 0.1, 0.1])

collisionShapeId = pb.createCollisionShape(
    shapeType=pb.GEOM_MESH,
    fileName=os.path.join(pb_data_path, 'random_urdfs/000/000.obj'),
    meshScale=[0.1, 0.1, 0.1])

# print("visualShapeId:", type(visualShapeId))
# print("collisionShapeId:", type(collisionShapeId))

multiBodyId = pb.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=collisionShapeId, 
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0, 0, 1],
    baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

print("multiBodyId:", multiBodyId)

texture_paths = glob.glob(os.path.join('/home/jiawei/Downloads/dtd', '**', '*.jpg'), recursive=True)
random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
textureId = pb.loadTexture(random_texture_path)
pb.changeVisualShape(multiBodyId, -1, textureUniqueId=textureId)

viewMatrix = pb.computeViewMatrix(
    cameraEyePosition=[0, 0, 5],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])

projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=5.1)

width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
    width=224, 
    height=224,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix)

pb.setGravity(0, 0, -9.8)
sim_freq = 240
time_stamp = 1./sim_freq
T = 100

start_t = time.time()
# pb.setRealTimeSimulation(1)

for i in range(T * sim_freq):
    soccer_pos, soccer_orn = pb.getBasePositionAndOrientation(soccer_ball)

    pb.applyExternalForce(soccer_ball,
                            linkIndex=-1,
                            forceObj=[-10, 0, 0],
                            posObj=soccer_pos,
                            flags=pb.WORLD_FRAME,
                            physicsClientId=physicsClient
                            )
    # t0 = time.time()
    pb.stepSimulation()
    # t1 = time.time()
    # print(f"step time: {t1 - t0} s")
    elapsed = time.time() - start_t
    
    if elapsed < i * time_stamp:
        time.sleep(i * time_stamp - elapsed)


pb.disconnect()


import numpy as np


# a = np.array([[2, 1, 3]])
# b = a[0]

# print(a)
# print(b)

# print(hex(id(a)))
# print(hex(id(b)))
# print(hex(id(a[0])))

# a[0] = np.array([3, 1, 2])

# print(a)
# print(b)