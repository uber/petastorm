# Developing Petastorm

Please follow the following instructions to develop Petastorm:

For tensorflow without GPU:
```virtualenv env
source env/bin/activate
pip install -U pip
pip install -e .[opencv,tf,test,torch]
```

For tensorflow with GPU:
```virtualenv env
source env/bin/activate
pip install -U pip
pip install -e .[opencv,tf_gpu,test,torch]
```

To run tests, please run the following: ```pytest```
