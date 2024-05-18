from m5.objects import *
from m5.util import addToPath

addToPath("../")

import argparse

from common.CacheConfig import *
from common.Caches import *
from common.cpu2000 import *
from common.CpuConfig import *
from common.Options import *
from common.Simulation import *

parser = argparse.ArgumentParser()

addCommonOptions(parser)
addSEOptions(parser)

parser.add_argument(
    "-b",
    "--benchmark",
    default="",
    help="The SPEC2006 benchmark to run.Benchmarks should be separated by an underscore if multiple are to be run.",
)

options = parser.parse_args()

# def get_processes(options):
#     """Interprets provided options and returns a list of processes"""

#     multiprocesses = []
#     inputs = []
#     outputs = []
#     errouts = []
#     pargs = []

#     workloads = options.cmd.split(';')
#     if options.input != "":
#         inputs = options.input.split(';')
#     if options.output != "":
#         outputs = options.output.split(';')
#     if options.errout != "":
#         errouts = options.errout.split(';')
#     if options.options != "":
#         pargs = options.options.split(';')

#     idx = 0
#     for wrkld in workloads:
#         process = Process(pid = 100 + idx)
#         process.executable = wrkld
#         process.cwd = os.getcwd()

#         if options.env:
#             with open(options.env, 'r') as f:
#                 process.env = [line.rstrip() for line in f]

#         if len(pargs) > idx:
#             process.cmd = [wrkld] + pargs[idx].split()
#         else:
#             process.cmd = [wrkld]

#         if len(inputs) > idx:
#             process.input = inputs[idx]
#         if len(outputs) > idx:
#             process.output = outputs[idx]
#         if len(errouts) > idx:
#             process.errout = errouts[idx]

#         multiprocesses.append(process)
#         idx += 1

#     if options.smt:
#         assert(options.cpu_type == "DerivO3CPU")
#         return multiprocesses, idx
#     else:
#         return multiprocesses, 1


def get_benchmark(options):
    """Interprets provided options and returns a list of processes"""

    multiprocesses = []

    # parsed by underscore
    # ex. gcc_gcc_gcc_gcc
    workloads = options.benchmark.split("_")
    x86_suffix = "_base.amd64-m64-gcc42-nn"

    suffix = "./benchs/"
    bench_path = "./spec2006/"

    idx = 0
    for wrkld in workloads:
        process = Process(pid=100 + idx)
        # process = LiveProcess()

        process.cwd = "./spec2006"

        # To add the directory of benchmarks as a prefix
        # flags in the below must be also added
        if wrkld == "alexnet":
            process.executable = suffix + wrkld
            process.cmd = [process.executable]
            # print("debugging JH", process.cmd)
        elif wrkld == "squeezenet":
            process.executable = suffix + wrkld
            process.cmd = [process.executable]
            # print("debugging JH", process.cmd)
        elif wrkld == "googlenet":
            process.executable = suffix + wrkld
            process.cmd = [process.executable]
            # print("debugging JH", process.cmd)
        elif wrkld == "lstm":
            process.executable = suffix + wrkld
            process.cmd = [process.executable]
            # print("debugging JH", process.cmd)
        if wrkld == "perlbench":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["-I.", "-I./lib", "attrs.pl"]
        elif wrkld == "bzip2":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["input.program", "5"]
        elif wrkld == "gcc":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["cccp.i", "-o", "cccp.s"]
        elif wrkld == "bwaves":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable]
        elif wrkld == "mcf":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["inp.in"]
        elif wrkld == "milc":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable]
            process.input = "su3imp.in"
        elif wrkld == "cactusADM":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["benchADM.par"]
        elif wrkld == "leslie3d":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable]
            process.input = "leslie3d.in"
        elif wrkld == "soplex":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["-m10000", "test.mps"]
        elif wrkld == "hmmer":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + [
                "--fixed",
                "0",
                "--mean",
                "325",
                "--num",
                "45000",
                "--sd",
                "200",
                "--seed",
                "0",
                "bombesin.hmm",
            ]
        elif wrkld == "GemsFDTD":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable]
        elif wrkld == "libquantum":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["33", "5"]
        elif wrkld == "h264ref":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + [
                "-d",
                "foreman_test_encoder_baseline.cfg",
            ]
        elif wrkld == "tonto":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable]
        elif wrkld == "lbm":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + [
                "20",
                "reference.dat",
                "0",
                "1",
                "100_100_130_cf_a.of",
            ]
        elif wrkld == "omnetpp":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["omnetpp.ini"]
        elif wrkld == "astar":
            process.executable = bench_path + wrkld + x86_suffix
            process.cmd = [process.executable] + ["lake.cfg"]

        idx += 1
        multiprocesses.append(process)

    # print("   JWL SE DEBUG: ", multiprocesses)
    return multiprocesses, 1


multiprocesses = []
numThreads = 1

# if options.bench:
#     apps = options.bench.split("-")
#     if len(apps) != options.num_cpus:
#         print("number of benchmarks not equal to set num_cpus!")
#         sys.exit(1)

#     for app in apps:
#         try:
#             if buildEnv['TARGET_ISA'] == 'alpha':
#                 exec("workload = %s('alpha', 'tru64', '%s')" % (
#                         app, options.spec_input))
#             elif buildEnv['TARGET_ISA'] == 'arm':
#                 exec("workload = %s('arm_%s', 'linux', '%s')" % (
#                         app, options.arm_iset, options.spec_input))
#             else:
#                 exec("workload = %s(buildEnv['TARGET_ISA', 'linux', '%s')" % (
#                         app, options.spec_input))
#             multiprocesses.append(workload.makeProcess())
#         except:
#             print("Unable to find workload for %s: %s" %
#                   (buildEnv['TARGET_ISA'], app),
#                   file=sys.stderr)
#             sys.exit(1)
# elif options.cmd:
#     multiprocesses, numThreads = get_processes(options)

if options.benchmark:
    multiprocesses, numThreads = get_benchmark(options)
    # print("  JWL SE DEBUG: after get_benchmark() ", multiprocesses[0].cmd)
else:
    print("No workload specified. Exiting!\n", file=sys.stderr)
    sys.exit(1)


system = System()

system.voltage_domain = VoltageDomain(voltage=options.sys_voltage)
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = options.sys_clock  # 1GHz Defualt
system.clk_domain.voltage_domain = system.voltage_domain

(CPUClass, test_mem_mode, FutureClass) = setCPUClass(options)
CPUClass.numThreads = numThreads

# Check -- do not allow SMT with multiple CPUs
if options.num_cpus > 1 and options.smt:
    fatal("SMT with multiple CPUs is not supported.")

np = options.num_cpus
system.cpu = [CPUClass(cpu_id=i) for i in range(np)]

system.cpu_voltage_domain = VoltageDomain()
system.cpu_clk_domain = SrcClockDomain()
system.cpu_clk_domain.clock = options.cpu_clock  # 2GHz Default
system.cpu_clk_domain.voltage_domain = system.cpu_voltage_domain

# All cpus belong to a common cpu_clk_domain, therefore running at a common
# frequency.
for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

if numThreads > 1:
    system.multi_thread = True

system.mem_mode = test_mem_mode
system.mem_ranges = [AddrRange(options.mem_size)]
system.cache_line_size = options.cacheline_size

system.membus = SystemXBar()

fast_mem_ctrl = MemCtrl()
fast_mem_ctrl.dram = DDR3_1600_8x8()
fast_mem_ctrl.dram.tCL = "40ns"  # Column Access Strobe latency
fast_mem_ctrl.dram.tRCD = "40ns"  # Row to Column Delay
fast_mem_ctrl.dram.tRP = "40ns"  # Row Precharge
fast_mem_ctrl.dram.tRAS = "80ns"  # Row Active Time
fast_mem_ctrl.boundary = options.fast_mem_size
fast_mem_ctrl.dram.range = AddrRange("0GB", options.mem_size)
fast_mem_ctrl.port = system.membus.mem_side_ports

# slow_mem_ctrl = MemCtrl()
# slow_mem_ctrl.dram = DDR3_1600_8x8()
# slow_mem_ctrl.dram.tCL = "40ns"
# slow_mem_ctrl.dram.tRCD = "40ns"
# slow_mem_ctrl.dram.tRP = "40ns"
# slow_mem_ctrl.dram.tRAS = "80ns"
# slow_mem_ctrl.page_migration_overhead = "1000ns"
# slow_mem_ctrl.cxl_additional_latency = "120ns"
# slow_mem_ctrl.boundary = options.fast_mem_size
# slow_mem_ctrl.dram.range = AddrRange(options.fast_mem_size, options.mem_size)
# slow_mem_ctrl.port = system.membus.mem_side_ports

system.mem_ctrls = fast_mem_ctrl

system.system_port = system.membus.cpu_side_ports

system = config_cache(options, system)

for i in range(np):
    if options.smt:
        system.cpu[i].workload = multiprocesses
    elif len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]
    system.cpu[i].max_insts_any_thread = options.maxinsts / np
    system.cpu[i].createThreads()

# workload for each CPU is same
system.workload = SEWorkload.init_compatible(multiprocesses[0].executable)

root = Root(full_system=False, system=system)

m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
