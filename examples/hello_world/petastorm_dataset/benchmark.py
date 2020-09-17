import time
import tensorflow as tf
from petastorm import make_reader
from petastorm.tf_utils import make_petastorm_dataset, tf_tensors

input_path ='file:///tmp/hello_world_dataset'
# mode = 'tf_dataset'
tic1 = time.time()

tf.enable_eager_execution()


for mode in (
        'tf_dataset',
        'python',
        'tf_tensors',):  #
    for workers_type in (
            'thread',
            'process'
    ):
        for workers_count in (
                1,
                5,
                10,
                20, 30, 40
        ):
            total_number_of_files = 0
            with make_reader(input_path, workers_count=workers_count, reader_pool_type=workers_type, num_epochs=20) as reader:
                if mode == 'python':
                    # Pure python
                    tic2 = time.time()
                    for row in reader:
                        total_number_of_files += 1
                        frame = row.frame
                        annotations = row.annotations

                if mode=='tf_dataset':
                    # Tensorflow tf.data.Dataset API
                    dataset = make_petastorm_dataset(reader)
                    tic2 = time.time()

                    for tensors in dataset:
                        total_number_of_files += 1

                if mode=='tf_tensors':
                    with tf.Session() as sess:
                        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                        try:
                            row_tensors = tf_tensors(reader)
                            tic2 = time.time()
                            while True:
                                sample = sess.run(row_tensors)
                                total_number_of_files += 1
                        except:
                            pass


            tic3 = time.time()
            print(f"{workers_type}-{mode} {workers_count} {(tic3 - tic2)/total_number_of_files} {total_number_of_files/(tic3 - tic2)}")