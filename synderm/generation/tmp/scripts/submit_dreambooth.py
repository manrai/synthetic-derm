"""
Generic submitit script
"""
import os
import datetime
import shlex
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from textwrap import dedent

import submitit
from tap import Tap


CLASS_NAMES = [
    "allergic-contact-dermatitis",
    "basal-cell-carcinoma",
    "folliculitis",
    "lichen-planus",
    "lupus-erythematosus",
    "neutrophilic-dermatoses",
    "photodermatoses",
    "psoriasis",
    "sarcoidosis",
    "squamous-cell-carcinoma",
    # "all"  # remove all for now
]


class SubmitItArgs(Tap):
    debug: bool = False
    submit: bool = False
    class_name: Optional[str] = None


# Args
args = SubmitItArgs().parse_args()
DEBUG = args.debug
SUBMIT = args.submit

# Check
if args.class_name is None:
    class_names = CLASS_NAMES
elif args.class_name in CLASS_NAMES:
    class_names = [args.class_name]
else:
    raise ValueError(args.class_name)

# Path
data_root = Path(os.getenv("DERM_ROOT", "/n/data1/hms/dbmi/manrai/derm"))
data_dir = data_root / 'Fitzpatrick17k'
# if not data_dir.is_dir():
#     raise ValueError(str(data_dir))

# Shared command args
train_args_str = dedent(f"""\
--instance_data_dir={str(data_dir)} \
--resolution=512 \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--learning_rate=5e-06 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_train_epochs=4 \
--report_to="wandb" """)

sample_args_str = dedent(f"""\

""")

# Commands
commands = []
for class_name in class_names:
    prompt = 'An image of {}, a skin disease'
    validation_prompt = prompt.format(class_name.replace('-', ' ') if class_name != 'all' else 'psoriasis')

    ########### Train ###########

    # # Text-to-image
    # output_dir = f'dreambooth-outputs/{class_name}'
    # command = dedent(f"""python train_dreambooth.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --instance_prompt="{prompt}" --validation_prompt="{validation_prompt}" --output_dir="{output_dir}" --disease_class={class_name} {train_args_str}""")
    # commands.append(command)

    # # Text-to-image with Fitzpatrick scale in prompt
    # output_dir = f'dreambooth-with-fitzpatrick-outputs/{class_name}'
    # command = dedent(f"""python train_dreambooth.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --instance_prompt="{prompt}" --validation_prompt="{validation_prompt}" --output_dir="{output_dir}" --disease_class={class_name} {train_args_str} --add_fitzpatrick_scale_to_prompt""")
    # commands.append(command)

    # # Inpaint
    # output_dir = f'dreambooth-inpaint-outputs/{class_name}'
    # command = dedent(f"""python train_dreambooth_inpaint.py --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" --instance_prompt="{prompt}" --validation_prompt="{validation_prompt}" --output_dir="{output_dir}" --disease_class={class_name} {train_args_str}""")
    # commands.append(command)

    ########### Sample ###########

    # # Text-to-image
    # checkpoint_dir = f'dreambooth-outputs/{class_name}'
    # command = dedent(f"""python generate.py --instance_data_dir={str(data_dir)} --model_type "text-to-image" --pretrained_model_name_or_path={checkpoint_dir} --instance_prompt="{prompt}" --disease_class={class_name}""")
    # commands.append(command)

    # # Inpaint
    # checkpoint_dir = f'dreambooth-inpaint-outputs/{class_name}'
    # command = dedent(f"""python generate.py --instance_data_dir={str(data_dir)} --model_type "inpaint" --pretrained_model_name_or_path={checkpoint_dir} --instance_prompt="{prompt}" --disease_class={class_name}""")
    # commands.append(command)

    # # To continue, add --start_index 10
    # checkpoint_dir = f'dreambooth-outputs/{class_name}'
    # command = dedent(f"""python generate.py --instance_data_dir={str(data_dir)} --model_type "text-to-image" --pretrained_model_name_or_path={checkpoint_dir} --instance_prompt="{prompt}" --disease_class={class_name} --start_index 50 --num_generations_per_image 50 --batch_size 16 --output_root generations-more""")
    # commands.append(command)
    
    ########### Sample from pretrained models ###########

    if class_name != 'all':

        # Text-to-image
        command = dedent(f"""python generate.py --output_root generations-pretrained --instance_data_dir={str(data_dir)} --model_type "text-to-image" --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --instance_prompt="{prompt}" --disease_class={class_name}""")
        commands.append(command)

        # Inpaint
        command = dedent(f"""python generate.py --output_root generations-pretrained --instance_data_dir={str(data_dir)} --model_type "inpaint" --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" --instance_prompt="{prompt}" --disease_class={class_name}""")
        commands.append(command)


if args.debug:
    commands = commands[-10:]

# Create executor
Path("slurm_logs").mkdir(exist_ok=True)
executor = submitit.AutoExecutor(folder="slurm_logs")
executor.update_parameters(
    tasks_per_node=1, nodes=1,
    timeout_min=23 * 60,
    slurm_partition="gpu_quad",  # devaccel learnaccel scavenge
    slurm_gres="gpu:1",
    # slurm_constraint="volta32gb",
    slurm_job_name="submititjob",
    cpus_per_task=16,
    mem_gb=48.0,
    # mail_type="END,FAIL",  # mail_user="lukemk@robots.ox.ac.uk"
)

# Context
if args.submit:
    print('Start')
    print(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    start_time = time.time()
    context = executor.batch
else:
    context = nullcontext

# Submitit via SLURM array
jobs = []
with context():
    for command in commands:
        print(command)
        function = submitit.helpers.CommandFunction(shlex.split(command))
        if args.submit:
            job = executor.submit(function)
            jobs.append(job)

if args.submit:

    # Print immediately
    print(f'Submitting: {len(jobs)} jobs')
    time.sleep(8)
    print(f'Submitted: {len(jobs)} jobs')
    print(f'Finished instantly: {sum(job.done() for job in jobs)} jobs')
    
    # Then wait until all jobs are completed:
    outputs = [job.result() for job in jobs]
    print(f'Finished all ({len(outputs)}) jobs in {time.time() - start_time:.1f} seconds')
    print(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))

