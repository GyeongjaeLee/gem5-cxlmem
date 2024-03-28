#03.28

from m5.objects import *
from common.CacheConfig import *

import argparse

parser = argparse.ArgumentParser()

addNoISAOptions(parser)

options = parser.parse_args()

system = System()

system.clk_domain = SrcClockDomain()
system.clk_domain.clock = options.sys_clock
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('16GB')]

system.cpu = X86TimingSimpleCPU()
system.membus = SystemXBar()

fast_mem_ctrl = MemCtrl()
fast_mem_ctrl.dram = DDR3_1600_8x8
fast_mem_ctrl.dram.tCL = '20ns'  # Column Access Strobe latency
fast_mem_ctrl.dram.tRCD = '20ns'  # Row to Column Delay
fast_mem_ctrl.dram.tRP = '20ns'  # Row Precharge
fast_mem_ctrl.dram.tRAS = '40ns'  # Row Active Time
fast_mem_ctrl.dram.range = AddrRange('0GB', '8GB')
fast_mem_ctrl.port = sytem.membus.master

slow_mem_ctrl = MemCtrl()
slow_mem_ctrl.dram = DDR3_1600_8x8
slow_mem_ctrl.dram.tCL = '40ns'
slow_mem_ctrl.dram.tRCD = '40ns'
slow_mem_ctrl.dram.tRP = '40ns'
slow_mem_ctrl.dram.tRAS = '80ns'
slow_mem_ctrl.dram.range = AddrRange('8GB', '16GB')
slow_mem_ctrl.port = system.membus.master

system.mem_ctrls = [fast_mem_ctrl, slow_mem_ctrl]

system.system_port = system.membus.slave

system = config_cache(options, system)
