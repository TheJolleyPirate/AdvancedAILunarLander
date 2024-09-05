## Preparation
Upgrade pip with:
`pip3 install --upgrade pip`

## Installation
```commandline
pip install swig
pip install "gymnasium[box2d]"
pip install "stable_baselines3[extra]"
```


## Debugging

### AttributeError
If you come to error like:
`AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'`
try to run
```commandline
pip3 install box2d box2d-kengz
```
Or try
```commandline
pip unistall box2d-py
pip install "gymnasium[box2d]"
```
