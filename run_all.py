import os
import subprocess

# Folder where scripts are located
script_dir = "/home/azwad/Works/Deep_Learning/Implementation_Phase/KD_Training_scripts"

# Get all Python scripts in the folder
scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])

# Run each script sequentially
for script in scripts:
    script_path = os.path.join(script_dir, script)
    print(f"Running {script}...")
    subprocess.run(["python", script_path])
    print(f"{script} finished!\n")