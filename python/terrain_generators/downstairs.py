import numpy as np

base_str = '''
<?xml version="1.0"?>
<robot name="downstairs_terrain">
    <link name="downstairs_terrain">
        {}
    </link>
</robot>
'''

init_str = ''
# head_stage = '''
#     <visual>
#       <origin rpy="0 0 0" xyz="5 -0.5 0 "/>
#       <geometry>
#         <box size="15 0.05 3.0"/>
#       </geometry>
#        <material name="white">
#         <color rgba="0.5 0.5 0.5 1.0"/>
#       </material>
#     </visual>
#     <collision>
#       <origin rpy="0 0 0" xyz="5 -0.5 0 "/>
#       <geometry>
#         <box size="15 0.05 3.0"/>
#       </geometry>
#     </collision>

#     <visual>
#       <origin rpy="0 0 0" xyz="5 0.5 0 "/>
#       <geometry>
#         <box size="15 0.05 3.0"/>
#       </geometry>
#        <material name="white">
#         <color rgba="0.5 0.5 0.5 1.0"/>
#       </material>
#     </visual>
#     <collision>
#       <origin rpy="0 0 0" xyz="5 0.5 0 "/>
#       <geometry>
#         <box size="15 0.05 3.0"/>
#       </geometry>
#     </collision>
# '''
# init_str += head_stage

base_interval = 0.01
current_height = 4
i = 0
while current_height > 0.02:
    i += 1
    pos = [0.3 * i + base_interval, 0]
    aug_str = '''
        <visual>
        <origin xyz="{} {} 0.05"/>
        <geometry>
            <box size="0.3 1.0 {}"/>
        </geometry>
        </visual>
        <collision>
        <origin xyz="{} {} 0.05"/>
        <geometry>
            <box size="0.3 1.0 {}"/>
        </geometry>
        <material name="white">
            <color rgba="0.5 0.5 0.5 1.0"/>
        </material>
        </collision>
        <inertial>
        <density value="567.0"/>
        </inertial>
    '''.format(pos[0], pos[1], current_height, pos[0], pos[1], current_height)

    init_str += aug_str
    current_height -= (0.05 + 0.011 * i)

base_str = base_str.format(init_str)

with open("downstairs.urdf", 'w+') as f:
    f.write(base_str)
