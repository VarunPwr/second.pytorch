import numpy as np

base_str = '''
<?xml version="1.0"?>
<robot name="curriculum_way_terrain">
    <link name="curriculum_way_terrain">
        {}
    </link>
</robot>
'''

init_str = ''

head_stage = '''
    <visual>
    <origin xyz="-2 0 0.05"/>
    <geometry>
        <box size="4 0.7 0.5"/>
    </geometry>
    </visual>
    <collision>
    <origin xyz="-2 0 0.05"/>
    <geometry>
        <box size="4 0.7 0.5"/>
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

extra_length = 1
sum_length = 0
for i in range(10):
    for j in range(2):
        pos = [0.29 * i + sum_length, -0.1 + 0.2 * j]
        for k in range(2):
            pos[k] += np.random.uniform(-0.02, 0.02)
        if j % 2 == 1:
            pos[1] += 0.1
        length = 0.14 + extra_length - 0.1 * i
        if i == 1:
            if j % 2 == 1:
                length += 0.22
        if i == 2:
            if j % 2 == 0:
                length += 0.22
        aug_str = '''
            <visual>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="{} 0.12 0.5"/>
            </geometry>
            </visual>
            <collision>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="{} 0.12 0.5"/>
            </geometry>
            <material name="white">
                <color rgba="0.5 0.5 0.5 1.0"/>
            </material>
            </collision>
            <inertial>
            <density value="567.0"/>
            </inertial>
        '''.format(pos[0], pos[1], length, pos[0], pos[1], length)

        init_str += aug_str
    sum_length += (extra_length - 0.1 * i)
base_str = base_str.format(init_str)

with open("curriculum_way.urdf", 'w+') as f:
    f.write(base_str)
