import numpy as np

base_str = '''
<?xml version="1.0"?>
<robot name="jumping_stages_terrain">
    <link name="jumping_stages_terrain">
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

base_interval = 0.15

for i in range(10):
    pos = [1.2 * i + (i + 1) * base_interval, 0]
    aug_str = '''
        <visual>
        <origin xyz="{} {} 0.05"/>
        <geometry>
            <box size="1.2 0.6 0.5"/>
        </geometry>
        </visual>
        <collision>
        <origin xyz="{} {} 0.05"/>
        <geometry>
            <box size="1.2 0.6 0.5"/>
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

with open("jumping_stages.urdf", 'w+') as f:
    f.write(base_str)
