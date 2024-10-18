# This script requires pv.

# https://www.ivarch.com/programs/pv.shtml

# There is a Homebrew formula to install pv on macOS.

# This script also depends on the simulation data:
# make -C data

import subprocess
import sys


def run_time_pv(command: str):
    print(command)
    subprocess.run(f"time {command} | pv > /dev/null", shell=True)
    print()


def run_bcftools(command: str, dataset_name: str):
    run_time_pv(f"bcftools {command} data/{dataset_name}.vcf.gz")


def run_vcztools(command: str, dataset_name: str):
    run_time_pv(f"vcztools {command} data/{dataset_name}.vcz")


if __name__ == "__main__":
    commands = [
        ("view", "sim_10k"),
        ("view", "chr22"),
        ("view -s tsk_7068,tsk_8769,tsk_8820", "sim_10k"),
        (r"query -f '%CHROM %POS %REF %ALT{0}\n'", "sim_10k"),
        (r"query -f '%CHROM:%POS\n' -i 'POS=49887394 | POS=50816415'", "sim_10k"),
        ("view -s '' --force-samples", "sim_10k"),
        ("view -i 'FMT/DP>10 & FMT/GQ>10'", "chr22"),
        ("view -i 'QUAL>10 || FMT/GQ>10'", "chr22"),
        (r"query -f 'GQ:[ %GQ] \t GT:[ %GT]\n'", "chr22"),
    ]

    if len(sys.argv) == 2 and sys.argv[1].isnumeric():
        index = int(sys.argv[1])
        command, dataset = commands[index]
        run_bcftools(command, dataset)
        run_vcztools(command, dataset)
    else:
        for command, dataset in commands:
            run_bcftools(command, dataset)
            run_vcztools(command, dataset)
