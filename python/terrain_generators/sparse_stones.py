import numpy as np

base_str = '''
<?xml version="1.0"?>
<robot name="sparse_stones_terrain">
    <link name="sparse_stones_terrain">
        {}
    </link>
</robot>
'''

init_str = ''

head_stage = '''
    <visual>
    <origin xyz="-1 0 0.05"/>
    <geometry>
        <box size="2 0.7 0.5"/>
    </geometry>
    </visual>
    <collision>
    <origin xyz="-1 0 0.05"/>
    <geometry>
        <box size="2 0.7 0.5"/>
    </geometry>
    <material name="white">
        <color rgba="0.5 0.5 0.5 1.0"/>
    </material>
    </collision>
    <inertial>
    <density value="567.0"/>
    </inertial>
'''
init_str += head_stage

for i in range(20):
    for j in range(2):
        pos = [0.25 * i, -0.12 + 0.2 * j]
        for k in range(2):
            pos[k] += np.random.uniform(-0.02, 0.02)
        if j % 2 == 1:
            pos[1] += 0.1
        aug_str = '''
            <visual>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="0.1 0.1 0.5"/>
            </geometry>
            </visual>
            <collision>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="0.1 0.1 0.5"/>
            </geometry>
            <material name="white">
                <color rgba="0.5 0.5 0.5 1.0"/>
            </material>
            </collision>
            <inertial>
            <density value="567.0"/>
            </inertial>
        '''.format(pos[0], pos[1], pos[0], pos[1])

        init_str += aug_str

base_str = base_str.format(init_str)

with open("sparse_stones.urdf", 'w+') as f:
    f.write(base_str)
