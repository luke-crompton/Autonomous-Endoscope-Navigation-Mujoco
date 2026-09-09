import os
from glob import glob

from setuptools import setup

package_name = 'scope_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Luke Crompton',
    maintainer_email='lukecromptonn16@gmail.com',
    description='Deployment control loop for the autonomous-endoscope rig.',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # node executable  =  module : function
            'policy_node = scope_control.policy_node:main',
            'depth_bridge = scope_control.depth_bridge:main',
            'scope_link = scope_control.scope_link:main',
            'mock_scope_link = scope_control.mock_scope_link:main',
            'fake_depth_pub = scope_control.fake_depth_pub:main',
        ],
    },
)
