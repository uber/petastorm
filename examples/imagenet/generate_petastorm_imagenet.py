#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This utility converts a directory with files from the Imagenet dataset (http://image-net.org/) into a
petastorm dataset (Parquet format).

The script can run locally (use '--master=local[*]' command line argument), or submitted to a spark cluster.

Schema defined in examples.imagenet.schema.ImagenetSchema will be used. The schema

NOTE: Imagenet dataset needs to be requested and downloaded separately by the user.
"""
from __future__ import division

import argparse
import glob
import json
import os

import cv2
from pyspark.sql import SparkSession
from six.moves.urllib.request import urlopen  # pylint: disable=import-error

from examples.imagenet.schema import ImagenetSchema
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row


def _arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input-path', type=str, required=True,
                        help='Path to the imagenet directory. If you are running this script on a Spark cluster, '
                             'you should have this file be mounted and accessible to executors.')
    parser.add_argument('-o', '--output-url', type=str, required=True,
                        help='hdfs://... or file:/// url where the parquet dataset will be written to.')
    parser.add_argument('-m', '--master', type=str, required=False, default=None,
                        help='Spark master. Use --master=local[*] to run locally.')

    return parser


def download_nouns_mapping():
    """Downloads a mapping between noun id (``nXXXXXXXX`` form) and the noun string representation.

    :return: A dictionary: ``{noun_id : text}``
    """
    NOUN_MAP_URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    request = urlopen(NOUN_MAP_URL)
    class_map_json = request.read()

    # raw_dict has the form of {id: [noun_id, text]}. We flatten it into {noun_id: text}
    raw_dict = json.loads(class_map_json.decode("utf-8"))
    nouns_map = {k: v for k, v in raw_dict.values()}
    return nouns_map


def imagenet_directory_to_petastorm_dataset(imagenet_path, output_url, spark_master=None, parquet_files_count=100,
                                            noun_id_to_text=None):
    """Converts a directory with imagenet data into a petastorm dataset.

    Expected directory format is:

    >>> nXXXXXXXX/
    >>>    *.JPEG

    >>> nZZZZZZZZ/
    >>>    *.JPEG

    :param imagenet_path: a path to the directory containing ``n*/`` subdirectories. If you are running this script on
      a Spark cluster, you should have this file be mounted and accessible to executors.
    :param output_url: the location where your dataset will be written to. Should be a url: either
      ``file://...`` or ``hdfs://...``
    :param spark_master: A master parameter used by spark session builder. Use default value (``None``) to use system
      environment configured spark cluster. Use ``local[*]`` to run on a local box.
    :param noun_id_to_text: A dictionary: ``{noun_id : text}``. If ``None``, this function will download the dictionary
      from the Internet.
    :return: ``None``
    """
    session_builder = SparkSession \
        .builder \
        .appName('Imagenet Dataset Creation') \
        .config('spark.executor.memory', '10g') \
        .config('spark.driver.memory', '10g')  # Increase the memory if running locally with high number of executors
    if spark_master:
        session_builder.master(spark_master)

    spark = session_builder.getOrCreate()
    sc = spark.sparkContext

    # Get a list of noun_ids
    noun_ids = os.listdir(imagenet_path)
    if not all(noun_id.startswith('n') for noun_id in noun_ids):
        raise RuntimeError('Directory {} expected to contain only subdirectories with name '
                           'starting with "n".'.format(imagenet_path))

    if not noun_id_to_text:
        noun_id_to_text = download_nouns_mapping()

    ROWGROUP_SIZE_MB = 256
    with materialize_dataset(spark, output_url, ImagenetSchema, ROWGROUP_SIZE_MB):
        # list of [(nXXXX, 'noun-text'), ...]
        noun_id_text_list = map(lambda noun_id: (noun_id, noun_id_to_text[noun_id]), noun_ids)

        # rdd of [(nXXXX, 'noun-text', path), ...]
        noun_id_text_image_path_rdd = sc.parallelize(noun_id_text_list, min(len(noun_ids) / 10 + 1, 10000)) \
            .flatMap(lambda word_id_label: [word_id_label + (image_path,) for image_path in
                                            glob.glob(os.path.join(imagenet_path, word_id_label[0], '*.JPEG'))])

        # rdd of [(nXXXX, 'noun-text', image), ...]
        noun_id_text_image_rdd = noun_id_text_image_path_rdd \
            .map(lambda id_word_image_path:
                 {ImagenetSchema.noun_id.name: id_word_image_path[0],
                  ImagenetSchema.text.name: id_word_image_path[1],
                  ImagenetSchema.image.name: cv2.imread(id_word_image_path[2])})

        # Convert to pyspark.sql.Row
        sql_rows_rdd = noun_id_text_image_rdd.map(lambda r: dict_to_spark_row(ImagenetSchema, r))

        # Write out the result
        spark.createDataFrame(sql_rows_rdd, ImagenetSchema.as_spark_schema()) \
            .coalesce(parquet_files_count) \
            .write \
            .mode('overwrite') \
            .option('compression', 'none') \
            .parquet(output_url)


if __name__ == '__main__':
    args = _arg_parser().parse_args()
    imagenet_directory_to_petastorm_dataset(args.input_path, args.output_url)
