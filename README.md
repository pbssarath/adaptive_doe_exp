# Framework for performing Adaptive DoE with experiments

### Installation

The code uses PARyOpt to adaptively sample from a specified experimental space. 

``` bash
git clone https://github.com/pbssarath/adaptive_doe_exp.git
git submodule update --init --recursive
python3.7 -m venv venv
source venv/bin/activate.sh
pip install -r requirements.txt
```

This will set up PARyOpt as well as the codes for running adaptive DoE.

#### Updating

To update the code from the repository, go to the repository home directory 
`adaptive_doe_exp` and perform the following:

```bash
git pull
git submodule update --recursive
```



