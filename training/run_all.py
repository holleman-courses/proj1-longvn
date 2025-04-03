import subprocess

def run_script(script_name):
    print(f"Running {script_name}.py")
    result = subprocess.run(["python", f"{script_name}.py"])
    print("Done.\n")

run_script("project_model_eval")
run_script("to_tflite")
run_script("tflite_to_header")