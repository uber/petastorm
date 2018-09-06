# Troubleshooting

## Trouble running the pytorch example

### Segmentation fault using pytorch when reading dataset

As reported in uber/petastorm#52, the reader appears haphazardly to dump core
from segmentation fault.  @selitvin noted that these issues related to memory
allocation have also been seen with TF.  A temporary workaround is to use a
different memory allocator, such as `tcmalloc`, per below:

```bash
sudo apt-get install libtcmalloc-minimal4
LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4" python examples/mnist/pytorch_example.py
```

### Import error due to dlopen failing to load more object into static thread-local storage (TLS)

If you see the following error while trying to run the pytorch example, you are in luck:
```bash
  File "/usr/local/lib/python2.7/dist-packages/torch/__init__.py", line 80, in <module>
    from torch._C import *
ImportError: dlopen: cannot load any more object with static TLS
```

This problem stems from a known defect in glibc `dlopen` logic that made conservative
assumptions about static thread-local storage, specific with respect to surplus
DTV slots, which no longer suffice for modern compute needs.

Solutions, in increasing order of effort involved:
1. Do `import torch` as early as possible ([ref](https://github.com/tensorflow/models/issues/523#issuecomment-272754029))
2. Install `libgomp1` and `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1`
3. Build pytorch from source (see below, also [ref](https://github.com/pytorch/pytorch/issues/643))
4. Patch glibc to increase surplus DTV slots from 14 to 32 or 64.

For background, this issue was reported back in 2013 with pMatlab since 2012](https://stackoverflow.com/a/19468365).
For additional references, find the [glibc bug report and fix](https://sourceware.org/bugzilla/show_bug.cgi?id=17620),
and the accompanying Debian glibc bug report [#793689](https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=793689).
According to [#793641](https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=793641),
some variabnt of the static TLS fix was included in glibc-2.22.
The OpenMP library libgomp.so.1 has had this fix in place since circa 2015.
Ubuntu Xenial and above also contains this fix, but just updating operating system
may not be sufficient if `torch` links against its own version of glibc that
still uses static TLS.
An `ldd` analysis (per [#793689 common #20](https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=793689#24))
can reveal whether libraries like torch is __actually__ still using static TLS.

### Building pytorch from Dockerfile and using it

If you choose to build pytorch from source, you can do so using the
[pytorch Dockerfile](https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile) as follows:

1. Clone the pytorch repo and `docker build -t pytorch -f docker/pytorch/Dockerfile --build-arg PYTHON_VERSION=2.7.6 .`
    * Set `PYTHON_VERSION` to your version of choice, or leave out for pytorch Dockerfile default
2. Build the custom pytorch docker: `docker build -t petastorm_torch -f examples/mnist/pytorch/Dockerfile .`
3. Run the container and work with your code: `docker run -it --rm petastorm_torch:latest /bin/bash`
