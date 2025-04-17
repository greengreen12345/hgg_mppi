# Hindsight Goal Generation (HGG)




## Installation
First, clone the repo in your folder and create the conda environment. 
````bash
cd <project_folder>
git clone https://github.com/tud-amr/m3p2i-aip.git

conda create -n m3p2i-aip python=3.8
conda activate m3p2i-aip
````

This project requires the source code of IsaacGym. Check for the [prerequisites and troubleshooting](https://github.com/tud-amr/m3p2i-aip/blob/master/thirdparty/README.md). Download it from https://developer.nvidia.com/isaac-gym, unzip and paste it in the `thirdparty` folder. Move to IsaacGym and install the package.
````bash
cd <project_folder>/m3p2i-aip/thirdparty/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e. 
````

Then install the current package by:
````bash
cd <project_folder>/m3p2i-aip
pip install -e. 
````

## Running Commands

```bash
python train.py 

```
