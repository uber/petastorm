# Petastorm

[![Build Status](https://travis-ci.com/uber/petastorm.svg?branch=master)](https://travis-ci.com/uber/petastorm) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Petastorm is a library enabling the use of Parquet storage from Tensorflow, Pytorch and other Python based ML training frameworks. Using datasets stored in Parquet from a deep-learning training/evaluation procedures bridges between a common big-data storage format and deep-learning frameworks making it simpler to combine them.
## Install
Petastorm defines tensorflow or tensorflow-gpu as an “extra” package. That way you can choose to depend on either a CPU or a GPU version of tensorflow, or not to install it at all.

Note that although petastorm depends on opencv, it is defined as an extra package to allow opencv
to be installed by debian package (or source) as opposed to pip. It is recommended to install it when
testing out petastorm however (and is therefore shown in the examples).  

Install with tensorflow CPU dependency:
```bash
pip install petastorm[opencv, tf]
```

Install with tensorflow GPU dependency:
```bash
pip install petastorm[opencv, tf_gpu]
```
## Usage
### Generating a dataset
A dataset created using petastorm is stored in Parquet format. On top of Parquet schema, petastorm also stores higher level schema information that makes multidimensional arrays into a native part of a petastorm dataset. 

Petastorm also supports extensible data codecs. These enables a user to use one of the standard data compressions (jpeg, png) or implement her own.

Generating a dataset is done using pyspark. Pyspark natively supports Parquet format; easy to run on a single machine or on a Spark compute cluster. Here is a minimalistic example writing out a table with some random data.

```python
HelloWorldSchema = Unischema('HelloWorldSchema', [
   UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('image1', np.uint8, (128, 256, 3), CompressedImageCodec('png'), False),
   UnischemaField('other_data', np.uint8, (None, 128, 30, None), NdarrayCodec(), False),
])


def row_generator(x):
   """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
   return {'id': x,
           'image1': np.random.randint(0, 255, dtype=np.uint8, size=(128, 256, 3)),
           'other_data': np.random.randint(0, 255, dtype=np.uint8, size=(4, 128, 30, 3))}


def generate_hello_world_dataset(output_url='file:///tmp/hello_world_dataset'):
   rows_count = 10
   rowgroup_size_mb = 256

   spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
   sc = spark.sparkContext

   # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
   # well as save petastorm specific metadata
   with materialize_dataset(spark, output_url, HelloWorldSchema, rowgroup_size_mb):

       rows_rdd = sc.parallelize(range(rows_count))\
           .map(row_generator)\
           .map(lambda x: dict_to_spark_row(HelloWorldSchema, x))

       spark.createDataFrame(rows_rdd, HelloWorldSchema.as_spark_schema()) \
           .coalesce(10) \
           .write \
           .mode('overwrite') \
           .parquet(output_url)

```

- `HelloWorldSchema` is an instance of a `Unischema` object. `Unischema` is capable of rendering types of its fields into different framework specific formats, such as: Spark’s `StructType`, Tensorflow’s `tf.DType` and numpy’s `numpy.dtype`.
- To define a dataset field, you need to specify a type, shape, a codec instance and whether the field is nullable for each field of the Unischema.
- We use pyspark for writing output Parquet files. In this example, we launch pyspark on a local box (`.master('local[2]')`). Of course for a larger scale dataset generation we would need a real compute cluster.
- We wrap spark dataset generation code with `materialize_dataset` context manager. The context manager is responsible to configure row group size at the beginning and write out petastorm specific metadata at the end.
- The row generating code is expected to return a Python dictionary indexed by a field name. We use `row_generator` function for that. 
- `dict_to_spark_row` converts the dictionary into a `pyspark.Row` object while ensuring schema `HelloWorldSchema` compliance (shape, type and is-nullable condition are tested).
- Once we have a `pyspark.DataFrame` we write it out to a parquet storage. The parquet schema is automatically derives from `HelloWorldSchema`.

### Reading a dataset
Reading a dataset is simple using `petastorm.Reader` class:

```python
with Reader('hdfs://myhadoop/some_dataset') as reader:
   for row in reader:
       print(row)
```
`Reader` supports taking either a `hdfs://...` or `file://...` protocols.

Once a Reader is instantiated, you can use it as an iterator.
To hookup the reader into a tensorflow graph, you can use `tf_tensors` function:

```python
with Reader('file:///some/localpath/a_dataset') as reader:
   row_tensors = tf_tensors(reader)
   with tf.Session() as session:
       for _ in range(3):
           print(session.run(row_tensors))
```

TODO: Reading a dataset from pytorch

`petastorm.Reader` is the main entry point for a user code that accesses the data from an ML framework such as Tensorflow or Pytorch. The reader has multiple features such as:
- Selective column readout
- Multiple parallelism strategies: thread, process, single-threaded (for debug)
- N-grams readout support
- Row filtering (row predicates)
- Shuffling
- Partitioning for multi-GPU training
- Local caching

# Troubleshooting
See the [Troubleshooting](docs/troubleshooting.md) page and please submit the [ticket](https://github.com/uber/petastorm/issues/new)
if you can't find an answer.
