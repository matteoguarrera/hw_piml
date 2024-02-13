# do.py

# Import modules
import argparse
import os
import random
import subprocess
import string
from pathlib import Path

# Define global configurations
IMAGE_NAME = 'pi_ml_hw2'
USER_NAME = 'non_root_user'

# Define some general settings
expected_commands = {
    'build':  'Build an image based on a Dockerfile (that must be located in the current folder).',
    'run':    'Run the container of the image built from the Dockerfile in the current directory.',
    'remove': 'Remove the docker image.',
    'jupyter': 'Start the container and run a jupyter notebook within it.',
}

# Generate the help text for the 'command' argument
help_text_command = 'Meta command. Options: '
for command, description in expected_commands.items():
    help_text_command += f"({command}) {description}\n"

# Generate help text for additional attributes
help_text_verbose = "Specify if output should be 'verbose' when calling command 'build' (the output will be 'verbose' when this flag is passed.)."
help_text_no_cache = "Specify if no cached images should be used when calling command 'build' (no cached images will be used when this flag is passed.)."
help_text_port = "Specify the local port when calling command 'jupyter' (default is 8888)."
gpu_text_help = 'Specify which GPU to use. Default: 0'
help_text_shell = 'Bash command to execute in Docker container. None opens an interactive shell.'

# Only allow this file to be run as script
if __name__ != '__main__':
    err_msg = f"The Python script 'metadocker.py' can only be run from command line (as '__main__')!"
    raise ValueError(err_msg)

# Intialize the argument partser object
parser = argparse.ArgumentParser(description="Automate 'Docker' workflow.")
parser.add_argument('command', type=str, help=help_text_command)
parser.add_argument('--bash', type=str, help=help_text_shell, default=None)
parser.add_argument('--verbose', '-v', dest='verbose',
                    action='store_true', help=help_text_verbose)
parser.add_argument('--no-cache', '-nc', dest='no_cache',
                    action='store_true', help=help_text_no_cache)
parser.add_argument('--port', type=int, default=8888, help=help_text_port)
parser.add_argument('--gpu', type=int, default=0, help=gpu_text_help)

# Parse the arguments
args = parser.parse_args()


def exec_shell_command(shell_command):
    """ Execute a shell command as subprocess. """
    # Print the shell command
    print(shell_command)

    # Run the shell command as subprocess
    process = subprocess.Popen(shell_command, shell=True)

    # Collect the process status
    process.wait()


def generate_jupyter_token(length=20):
    """ Generate a jupyter token. """
    # Define the set of chars
    chars = string.ascii_letters + string.digits

    # Randomly sample from the set of chars to obtain a string of the requested length
    jupyter_token = ''.join(random.choice(chars) for _ in range(length))

    return jupyter_token


# Differ cases for the different commands
if args.command == 'build':  # Build image
    # Check that there exists a Dockerfile in the current directory
    if not os.path.isfile(Path(os.getcwd(), 'Dockerfile')):
        err_msg = f"Can't build image because there exists no Dockerfile in the current directory."
        raise FileNotFoundError(err_msg)

    # Use additional flags depending on passed arguments
    additional_flags = ''

    # If the output of 'build' should be 'verbose', add '--progress=plain' as additional flag.
    if args.verbose:
        additional_flags += '--progress=plain '

    # If the output of 'build' should not use cached images, add '--no-cache' as additional flag.
    if args.no_cache:
        additional_flags += '--no-cache '

    # Construct shell command to build the image
    docker_shell_command = 'sudo docker build ' + \
        additional_flags + ' -t ' + IMAGE_NAME + ' .'

    # Execute the shell command
    exec_shell_command(docker_shell_command)
elif args.command == 'remove':  # Remove image
    # Construct the shell command to remove the image
    docker_shell_command = 'sudo docker rmi ' + IMAGE_NAME

    # Execute the shell command
    exec_shell_command(docker_shell_command)

# Run the container in interactive mode and run command.
elif args.command == 'run':
    bash_cmd = args.bash if args.bash is not None else 'bash'
    docker_shell_command = 'sudo docker run -it --rm --runtime=nvidia --gpus device=' + \
        str(args.gpu) + ' -v "${PWD}":/home/' + USER_NAME + \
        '/project ' + ' -t ' + IMAGE_NAME + ' ' + bash_cmd
    print(docker_shell_command)
    processes = [
        subprocess.Popen(docker_shell_command, shell=True),
    ]

    # Collect process statuses
    exitcodes = [p.wait() for p in processes]

elif args.command == 'jupyter':  # Run container and start jupyter notebook environment
    # Extract the port from the arguments as string
    local_port = str(args.port)

    # Generate the jupyter token
    jupyter_token = generate_jupyter_token()

    # Construct the shell command to run the container
    # Remark: We provide the (local) port (of the machine), and the generated security token
    docker_shell_command = 'sudo docker run -it --rm -p ' + local_port + ':8888 -e JUPYTER_TOKEN=' + \
        jupyter_token + ' -v "${PWD}":/home/' + \
        USER_NAME + '/project ' + IMAGE_NAME + ' \"jupyter lab --ip=0.0.0.0\"'

    # Print the shell command
    print(docker_shell_command)

    # Define a shell command to open Jupyter Notebook in the default browser
    # Remark: 1) Wait some seconds before opening the Jupyter Notebook in the default browser.
    #         2) We need to pass the (local) port (of the machine) and the jupyter token.
    open_browser_shell_command = 'sleep 7; open http://127.0.0.1:' + \
        local_port + '/lab?token=' + jupyter_token

    # Run the subprocesses
    processes = [
        subprocess.Popen(docker_shell_command, shell=True),
        subprocess.Popen(open_browser_shell_command, shell=True),
    ]

    # Collect process statuses
    exitcodes = [p.wait() for p in processes]
elif args.command == 'stop':  # Stop all containers that derive from the image
    # Construct a shell command to stop all containers that derive from the image
    docker_shell_command = 'sudo docker stop $(docker ps -a -q --filter ancestor=' + \
        IMAGE_NAME + ' --format="{{.ID}}")'

    # Execute the shell command
    exec_shell_command(docker_shell_command)
    print(
        f"Stoped all containers that have been run from the image '{IMAGE_NAME}'")
else:
    err_msg = f"The first argument of 'do.py' must be a 'command' corresponding to one of the following:\n{list(expected_commands)}."
    raise ValueError(err_msg)
