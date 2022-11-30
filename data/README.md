# Dataset

### Data Format
The SDF samples for each mesh is stored in an *N*x4 numpy array file (.npy).<br>
Each of the *N* rows contain a 3D coordiate and an SDF value.

|           | X Position | Y Position | Z Position | SDF Value  |
| --------- | ---------- | ---------- | ---------- | ---------- |
| Sample 1  |            |            |            |            |
| Sample 2  |            |            |            |            |
| ...       |            |            |            |            |
| Sample N  |            |            |            |            |


### Process a 3D Mesh
The steps to prepare a 3D mesh are:
1. Scale the raw 3D mesh to fit a unit sphere.
2. Generate a uniform point cloud in the unit sphere. 100,000 points or more is recommended.
3. Compute the distance from each point to the surface of the mesh.

This processing can be done with the help of the **[mesh-to-sdf](https://pypi.org/project/mesh-to-sdf/)** python package.<br>
<br>
**Example Code:**
```python
import numpy
import trimesh
import mesh_to_sdf
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from mesh_to_sdf.utils import scale_to_unit_sphere

mesh = trimesh.load('chair.obj')
mesh = scale_to_unit_sphere(mesh)

points = sample_uniform_points_in_unit_sphere(100000)
sdf = mesh_to_sdf.mesh_to_sdf(mesh, points)
combined = numpy.concatenate((points, numpy.expand_dims(sdf, axis=1)), 1)

numpy.save('chair_sample.npy', combined)
```


### Process the ShapeNet Dataset
To processes the ShapeNet dataset from scratch, you can use the **[prepare_shapenet_dataset.py](https://github.com/marian42/shapegan/blob/master/prepare_shapenet_dataset.py)** utility from the **[ShapeGAN](https://github.com/marian42/shapegan)** GitHub project.<br>
Alternatively, the authors provide a **[preprocessed dataset](https://ls7-data.cs.tu-dortmund.de/shape_net/ShapeNet_SDF.tar.gz)** with SDF samples for the airplane, chair, and sofa categories of the ShapeNet dataset.