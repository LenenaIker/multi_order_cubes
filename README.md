# Multi-Order Cubes (Isaac Lab + SAC)

Custom robotic manipulation environment built with **Isaac Lab** for training a **Soft Actor-Critic (SAC)** agent on a conditional pick-and-place task.

---

## 1. Overview

This project defines a structured manipulation environment where:

* There are **4 horizontal discrete positions**
* **3 positions contain cubes**
* **1 position is empty**
* Each cube has:

  * A different size
  * A different shade of blue (light-blue, blue, dark-blue)

The agent receives two additional inputs:

* `from`: source position
* `to`: target position

Example:

```
from = 2
to   = 4
```

The robot must pick the cube from position 2 and place it at position 4.

The reward function evaluates whether the requested instruction has been correctly executed.

---

## 2. Tech Stack

System configuration :

### Software

* Ubuntu 24
* Isaac Sim 5.1.0
* Isaac Lab 2.3.1

---

## 3. Installation

The full installation procedure (NVIDIA drivers, Vulkan validation, Isaac Sim, Miniforge, Isaac Lab, Conda environment setup) is documented in:



### High-Level Steps

1. Install NVIDIA driver 580.xx
2. Install Isaac Sim 5.1.0 under `/opt/isaacsim`
3. Create Conda environment (Python 3.11)
4. Clone IsaacLab v2.3.1
5. Create symbolic link to Isaac Sim
6. Inject Isaac Sim runtime:

```bash
source _isaac_sim/setup_conda_env.sh
```

7. Run official tutorials to verify installation

---

## 4. Environment Design

Specification derived from project objectives .

### Scene Layout

* 4 discrete horizontal positions
* 3 cubes:

  * Different sizes
  * Different blue shades
* 1 empty slot

### Observations

* Physical state of the environment
* Cube positions
* Instruction inputs (`from`, `to`)

### Actions

* Continuous control of robotic arm
* Pick-and-place manipulation

### Reward Function

Positive reward when:

* The correct cube is picked
* It is placed in the requested target position

Penalties for:

* Incorrect cube
* Incorrect placement
* Collisions
* Failure to complete the task

---

## 5. Project Objective

Train a **SAC agent** that:

* Learns conditional manipulation (`from`, `to`)
* Generalizes across configurations
* Decouples size, color, and position
* Solves the task robustly
