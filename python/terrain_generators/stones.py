
base_str = '''
<?xml version="1.0"?>
<robot name="stones_terrain">
    <link name="stones_terrain">
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

for i in range(40):
    for j in range(20):
        pos = [0.15 * i, -1.5 + 0.15 * j]
        if i % 2 == 1:
            pos[1] += 0.06
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

# tail_stage = '''
#     <visual>
#     <origin xyz="7 0 0.05"/>
#     <geometry>
#         <box size="2 4 0.5"/>
#     </geometry>
#     </visual>
#     <collision>
#     <origin xyz="-1 0 0.05"/>
#     <geometry>
#         <box size="2 4 0.5"/>
#     </geometry>
#     <material name="white">
#         <color rgba="0.5 0.5 0.5 1.0"/>
#     </material>
#     </collision>
#     <inertial>
#     <density value="567.0"/>
#     </inertial>
# '''
# init_str += tail_stage

base_str = base_str.format(init_str)

with open("stones.urdf", 'w+') as f:
    f.write(base_str)
