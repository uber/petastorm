Examples
========

Hello world
-----------

.. automodule:: examples.hello_world.hello_world_dataset

.. automodule:: examples.hello_world.pytorch_hello_world

ImageNet
--------

.. automodule:: examples.imagenet.generate_petastorm_imagenet

MNist
-----

.. automodule:: examples.mnist.generate_petastorm_mnist

.. Pytorch is excluded for now because the free readthedocs sphinx doc build
   process limits build machine memory use to 1 GB, insufficient to import
   torch as part of the build process.  Should that ever change, include this:

   automodule:: examples.mnist.pytorch_example
