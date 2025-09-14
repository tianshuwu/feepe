gripper_name=$1
model_name=$2

blenderproc run src/renderer.py --gripper_name $gripper_name --model_name $model_name

