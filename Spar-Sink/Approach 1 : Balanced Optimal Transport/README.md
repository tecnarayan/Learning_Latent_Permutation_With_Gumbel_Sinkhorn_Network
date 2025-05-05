## Generating WFR Distance Matrices and MDS Plots

### Step 1: Generate WFR Distance Matrices

To generate the WFR distance matrices, run:

```bash
python3 echo_dist.py

Important:
Before running the script, manually change the variable inside the three modes in echo_dist.py to match the desired configuration.
Step 2: Generate MDS Plots

After generating the distance matrices, create the MDS plots by running:

python3 echo_mds.py

This will produce the MDS plots for the three cases.
Output Directory

    All output files will be saved in the output folder.

    This folder will be automatically created in the parent directory.

    You can modify the file paths in the scripts (echo_dist.py and echo_mds.py) as needed.
