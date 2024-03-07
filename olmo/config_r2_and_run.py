import sys
from os import mkdir, environ
from os.path import expanduser, join
from subprocess import run


def main():
    aws_dir = expanduser("~/.aws")
    print(f"Setting up AWS at {aws_dir}")
    mkdir(aws_dir)
    with open(join(aws_dir, "credentials"), "w") as f:
        f.write(environ["R2_CREDENTIALS"])
    print("Starting torchrun")
    run(["torchrun"] + list(sys.argv[1:]))


if __name__ == '__main__':
    main()
