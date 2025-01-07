# Lab 1: Fitting probability distributions

Welcome! Here are some setup tips to help get you started.

## Prerequisties
We have chosen to use a Jupyter notebook (previously iPython notebook),
for its nice didactic, easy-reading style. You can find installation
details [here](http://jupyter.readthedocs.io/en/latest/install.html), or
we provide our preferred installation method below.

You will want to have installed
```
python 3.6
jupyter
numpy
scipy
matplotlib
opencv
```

### Preferred setup
Visual Studio Code has very good debugging tools. You should have already created an environment from last week's lab with the instructions on moodle. Here they are again:
- Reboot the PC and select Windows (may not be essential, but if it works in Windows..)
- Download and install VS Code
- In VSCode, install the Python extension
- In VSCode, install the Jupyter extension
- Download and install MiniConda
- Open the Anaconda Prompt (AP) and use that to run commands you find in the "Conda Cheat Sheet" - just search for it online.
--> Ok, so in the lab, you may lack permissions to update all the existing packages - should still be ok.
- AP: conda create --name MV00
--> you could also specify a specific Python version, instead of the newest, e.g. python=3.9.7. You should be using a Python version around 3.7 or newer for our labs.
- AP: conda activate MV00
- AP: conda install numpy (as an example of a package you must install, before you can import that package into your code, e.g. for Part C of today's lab)
- AP: conda install jupyter (you'll need this so that VS Code can operate on notebook files)
- Close and re-run VS Code so it sees your new environment
- In VS Code, open any .ipynb file. Then in the upper right corner, select the Python environment that you created earlier (e.g. MV00).


### Alternatively, use notebooks directly:
1) Download anaconda: https://www.anaconda.com/download/
2) Create a new environment: `conda create -n machinevision python=3.7`
3) Enter that environment: `source activate machinevision`
4) Download packages: `conda install jupyter numpy scipy matplotlib opencv`
5) Launch Jupyter: `jupyter notebook`


### Troubleshooting
If upon installation you read the error `native kernel (python3) is not available`, 
then try:

```
conda install ipykernel
python -m ipykernel install --user
jupyter notebook &
```
