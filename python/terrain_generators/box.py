import numpy as np

base_str = '''
<?xml version="1.0"?>
<robot name="box_terrain">
    <link name="box_terrain">
        {}
    </link>
</robot>
'''

init_str = ''

for i in range(40):
    for j in range(20):
        pos = [-0.3 + 0.12 * i, -1 + 0.08 * j]
        for k in range(2):
            pos[k] += np.random.uniform(-0.05, 0.05)
        rand_height = np.random.uniform(0.02, 0.045)
        aug_str = '''
            <visual>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="0.06 0.06 {}"/>
            </geometry>
            </visual>
            <collision>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="0.06 0.06 {}"/>
            </geometry>
            <material name="white">
                <color rgba="0.5 0.5 0.5 1.0"/>
            </material>
            </collision>
            <inertial>
            <density value="567.0"/>
            </inertial>
        '''.format(pos[0], pos[1], rand_height, pos[0], pos[1], rand_height)

        init_str += aug_str

base_str = base_str.format(init_str)

with open("box_dense.urdf", 'w+') as f:
    f.write(base_str)
