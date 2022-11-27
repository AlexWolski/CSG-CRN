# Dataset
---
### Data Format
The SDF samples for each mesh is stored in an *N*x4 numpy array file (.npy).<br>
Each of the *N* rows contains contains a 3D coordiate and an SDF value.

|           | X Position | Y Position | Z Position | SDF Value  |
| --------- | ---------- | ---------- | ---------- | ---------- |
| Sample 1  |            |            |            |            |
| Sample 2  |            |            |            |            |
| ...       |            |            |            |            |
| Sample N  |            |            |            |            |


### Process New Meshes

To sample SDF values of new meshes, you can use the **[mesh-to-sdf](https://pypi.org/project/mesh-to-sdf/)** python package.


### Process ShapeNet Dataset

To processes the ShapeNet dataset from scratch, you can use the **[prepare_shapenet_dataset.py](https://github.com/marian42/shapegan/blob/master/prepare_shapenet_dataset.py)** utility from the **[ShapeGAN](https://github.com/marian42/shapegan)** GitHub project.<br>
Alternatively, the authors provide a **[preprocessed dataset](https://ls7-data.cs.tu-dortmund.de/shape_net/ShapeNet_SDF.tar.gz)** with SDF samples for the airplane, chair, and sofa categories of the ShapeNet dataset.