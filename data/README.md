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


### Create Custom Dataset

The **[prepare_sdf_dataset.py](https://github.com/AlexWolski/CSG-CRN/blob/master/prepare_sdf_dataset.py)** utility can be used to convert mesh files to SDF samples. This process relies on the **[mesh-to-sdf](https://pypi.org/project/mesh-to-sdf/)** python package and can quite slow.<br>
<br>
**Example Usage:**<br>
`python prepare_sdf_dataset.py [mesh_files_directory] [output_directory] [number_of_samples]`


### Process the ShapeNet Dataset
To processes the ShapeNet dataset from scratch, you can use the **[prepare_shapenet_dataset.py](https://github.com/marian42/shapegan/blob/master/prepare_shapenet_dataset.py)** utility from the **[ShapeGAN](https://github.com/marian42/shapegan)** GitHub project.<br>
<br>
Alternatively, you can download a preprocessed dataset. The ShapeGAN authors provide the **[ShapeNet SDF Dataset](https://ls7-data.cs.tu-dortmund.de/shape_net/ShapeNet_SDF.tar.gz)** (71GB) with SDF samples for the airplane, chair, and sofa categories of the ShapeNet dataset.<br>
Download links for the uniform samples of each shape class are available here:
* [Airplanes](https://huggingface.co/datasets/AlexWolski/ShapeNet-SDF-Uniform/resolve/main/airplanes.zip) (8.4GB)
* [Chairs](https://huggingface.co/datasets/AlexWolski/ShapeNet-SDF-Uniform/resolve/main/chairs.zip) (16.3GB)
* [Sofas](https://huggingface.co/datasets/AlexWolski/ShapeNet-SDF-Uniform/resolve/main/sofas.zip) (7.7GB)