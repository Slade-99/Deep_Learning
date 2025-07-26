import os
import subprocess
import sys

# Folder where scripts are located
script_dir = "/home/azwad/Works/HierNN/MTL_Soft_Gating/Train"

# Get all Python scripts in the folder
scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])

# Use the current Python interpreter
python_exec = sys.executable

# Run each script sequentially
for script in scripts:
    script_path = os.path.join(script_dir, script)
    print(f"Running {script}...")
    subprocess.run([python_exec, script_path])
    print(f"{script} finished!\n")