import numpy as np

base_str = '''
<?xml version="1.0"?>
<robot name="stone_terrain">
    <link name="stone_terrain">
        {}
    </link>
</robot>
'''

init_str = ''

for i in range(10):
    for j in range(5):
        pos = [-1 + 1 * i, -2 + 0.8 * j]
        for k in range(2):
            pos[k] += np.random.uniform(-0.05, 0.05)
        if i % 2 == 1:
            pos[1] += 0.5
        aug_str = '''
            <visual>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="0.9 0.6 0.5"/>
            </geometry>
            </visual>
            <collision>
            <origin xyz="{} {} 0.05"/>
            <geometry>
                <box size="0.9 0.6 0.5"/>
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

with open("stages.urdf", 'w+') as f:
    f.write(base_str)
