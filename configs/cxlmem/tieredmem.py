from m5.objects import *
from m5.util import addToPath

addToPath("../")

from common.CacheConfig import *
from common.Options import *


import argparse

parser = argparse.ArgumentParser()

addCommonOptions(parser)

options = parser.parse_args()

system = System()

system.clk_domain = SrcClockDomain()
system.clk_domain.clock = options.sys_clock # 1GHz Defualt
system.clk_domain.voltage_domain = VoltageDomain()

system.cpu_clk_domain = SrcClockDomain()
system.cpu_clk_domain.clock = options.cpu_clock # 2GHz Default
system.cpu_clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('16MB')]

system.cpu = X86TimingSimpleCPU()
system.membus = SystemXBar()

fast_mem_ctrl = MemCtrl()
fast_mem_ctrl.dram = DDR3_1600_8x8()
fast_mem_ctrl.dram.tCL = '20ns'  # Column Access Strobe latency
fast_mem_ctrl.dram.tRCD = '20ns'  # Row to Column Delay
fast_mem_ctrl.dram.tRP = '20ns'  # Row Precharge
fast_mem_ctrl.dram.tRAS = '40ns'  # Row Active Time
fast_mem_ctrl.dram.range = AddrRange('0GB', '8GB')
fast_mem_ctrl.port = system.membus.mem_side_ports

slow_mem_ctrl = MemCtrl()
slow_mem_ctrl.dram = DDR3_1600_8x8()
slow_mem_ctrl.dram.tCL = '40ns'
slow_mem_ctrl.dram.tRCD = '40ns'
slow_mem_ctrl.dram.tRP = '40ns'
slow_mem_ctrl.dram.tRAS = '80ns'
slow_mem_ctrl.dram.range = AddrRange('8GB', '16GB')
slow_mem_ctrl.port = system.membus.mem_side_ports

system.mem_ctrls = [fast_mem_ctrl, slow_mem_ctrl]

system.system_port = system.membus.cpu_side_ports

system = config_cache(options, system)

binary = 'tests/test-progs/threads/bin/x86/linux/threads'

system.workload = SEWorkload.init_compatible(binary)

process = Process()
process.cmd = [binary]
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system = False, system = system)
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()

print('Exiting @ tick {} because {}'.format(m5.curTick(), exit_event.getCause()))
