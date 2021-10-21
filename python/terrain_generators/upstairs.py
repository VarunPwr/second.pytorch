import numpy as np

base_str = '''
<?xml version="1.0"?>
<robot name="upstairs_terrain">
    <link name="upstairs_terrain">
        {}
    </link>
</robot>
'''

init_str = ''
head_stage = '''
    <visual>
      <origin rpy="0 0 0" xyz="5 -0.5 0 "/>
      <geometry>
        <box size="15 0.05 3.0"/>
      </geometry>
       <material name="white">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="5 -0.5 0 "/>
      <geometry>
        <box size="15 0.05 3.0"/>
      </geometry>
    </collision>

    <visual>
      <origin rpy="0 0 0" xyz="5 0.5 0 "/>
      <geometry>
        <box size="15 0.05 3.0"/>
      </geometry>
       <material name="white">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="5 0.5 0 "/>
      <geometry>
        <box size="15 0.05 3.0"/>
      </geometry>
    </collision>
'''
init_str += head_stage

base_interval = 0.01
current_height = 0
for i in range(30):
    pos = [0.4 * i + base_interval, 0]
    aug_str = '''
        <visual>
        <origin xyz="{} {} 0.05"/>
        <geometry>
            <box size="0.4 0.6 {}"/>
        </geometry>
        </visual>
        <collision>
        <origin xyz="{} {} 0.05"/>
        <geometry>
            <box size="0.4 0.6 {}"/>
        </geometry>
        <material name="white">
            <color rgba="0.5 0.5 0.5 1.0"/>
        </material>
        </collision>
        <inertial>
        <density value="567.0"/>
        </inertial>
    '''.format(pos[0], pos[1], current_height + (0.05 + 0.01 * i), pos[0], pos[1], current_height + (0.05 + + 0.01 * i))

    init_str += aug_str
    current_height += (0.05 + 0.008 * i)

base_str = base_str.format(init_str)

with open("upstairs.urdf", 'w+') as f:
    f.write(base_str)
