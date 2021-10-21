import numpy as np

base_str = '''
<?xml version="1.0"?>
<robot name="obstacles_terrain">
    <link name="obstacles_terrain">
        {}
    </link>
</robot>
'''

init_str = ''

for i in range(10):
    for j in range(8):
        pos = [2 + 0.8 * i, -4.8 + 1.2 * j]
        for k in range(2):
            pos[k] += np.random.uniform(-0.1, 0.1)
        if i % 2 == 1:
            pos[1] += 0.4
        rand_width = np.random.uniform(0.1, 0.2)
        aug_str = '''
            <visual>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="{} {} 1"/>
            </geometry>
            </visual>
            <collision>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="{} {} 1"/>
            </geometry>
            <material name="white">
                <color rgba="0.5 0.5 0.5 1.0"/>
            </material>
            </collision>
            <inertial>
            <density value="567.0"/>
            </inertial>
        '''.format(pos[0], pos[1], rand_width, rand_width, pos[0], pos[1], rand_width + 0.02, rand_width + 0.02)

        init_str += aug_str

base_str = base_str.format(init_str)

with open("obstacles.urdf", 'w+') as f:
    f.write(base_str)
