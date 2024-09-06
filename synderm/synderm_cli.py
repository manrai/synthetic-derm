import os
import click
import subprocess
import sys

@click.group()
def cli():
    pass

@cli.command()
@click.argument('script_name')
@click.argument('flags', nargs=-1)
def run(script_name, flags):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "generation", f"{script_name}.py")
    command = [sys.executable, script_path] + list(flags)
    subprocess.run(command)

if __name__ == '__main__':
    cli()