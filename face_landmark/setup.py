from setuptools import find_packages, setup

package_name = 'face_landmark'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'landmark_node = face_landmark.landmark_node:main',
            'face_node = face_landmark.face_node:main',  # 통합 노드 
            'recognition_node = face_landmark.face_recog_node:main',
            'detection_node = face_landmark.face_detec_node:main',
            'demo_face = face_landmark.demo_face:main',
        ],
    },
)
