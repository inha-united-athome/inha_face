<div align="center">
  <img src="https://github.com/inha-united-athome/.github/raw/main/profile/inha_logo.png" width="110" /><br/>
  <h1>inha_face</h1>
  <span><b>ROS 2 Face Recognition and Detection package</span><br/><br/>

  <img src="https://img.shields.io/badge/Ubuntu-22.04-orange?style=flat&logo=ubuntu" />
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/ROS2-Humble-blue?style=flat&logo=ros" />
  <br/><br/>

  <a href="#overview">Overview</a> • <a href="#architecture">Architecture</a> • <a href="#installation">Installation</a> • <a href="#usage">Usage</a> • <a href="#interfaces">Interfaces</a> • <a href="#contributing">Contributing</a>
</div>


Overview
---
`inha_face` provides face recognition and detection capabilities for the Inha-United@Home stack, enabling the robot to identify and track human faces during interactions.

Architecture
---
```text
inha_face/
├── 📁 launch/        # Launch files
├── 📁 config/        # Parameters and configs
├── 📁 src/           # Core nodes
├── 📁 scripts/       # Utilities
├── 📁 msg/           # Custom messages (if any)
├── 📁 srv/           # Custom services (if any)
└── 📄 README.md
```

Installation
---
**Prerequisites**
- Ubuntu 22.04
- ROS 2 Humble
- `colcon`, `rosdep`

```bash
mkdir -p ~/inha_ws/src
cd ~/inha_ws/src
git clone https://github.com/inha-united-athome/inha_face.git

cd ~/inha_ws
sudo rosdep init  # skip if already initialized
rosdep update
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
source install/setup.bash
```

Usage
---
```bash
# Run a node
ros2 run thor_whisper whisper_node
ros2 run thor_whisper tts_node
```

Interfaces ( **To Be Continue**)
---
**Topics**
| Name | Type | Description |
|------|------|-------------|
| `<topic_name>` | `<msg_type>` | `<what it does>` |

**Services**
| Name | Type | Description |
|------|------|-------------|
| `<service_name>` | `<srv_type>` | `<what it does>` |

**Actions (optional)**
| Name | Type | Description |
|------|------|-------------|
| `<action_name>` | `<action_type>` | `<what it does>` |

Contributing
---
- Create a branch: `git checkout -b feat/<name>`
- Commit: `git commit -m "Add <feature>"`
- Push: `git push origin feat/<name>`
- Open a Pull Request
