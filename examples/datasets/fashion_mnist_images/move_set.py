import os
import subprocess


def main():
    os.system("mkdir predict")
    for i in range(10):
        os.system(f"mkdir predict/{i}")
        num = int(subprocess.check_output(f"ls ./test/{i}/ | wc -l", shell=True))
        for j in range(100):
            os.system(f"mv ./test/{i}/0{num-j-1}.png ./predict/{i}/0{num-j-1}.png")


if __name__ == "__main__":
    main()
