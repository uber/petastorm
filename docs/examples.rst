Examples
========

Hello World
-----------

.. automodule:: examples.hello_world.hello_world_dataset

ImageNet
--------

.. automodule:: examples.imagenet.generate_petastorm_imagenet

MNIST
-----

.. automodule:: examples.mnist.generate_petastorm_mnist

.. Pytorch is excluded for now because the free readthedocs sphinx doc build
   process limits build machine memory use to 1 GB, insufficient to import
   torch as part of the build process.  Should that ever change, include this:

   automodule:: examples.mnist.main
