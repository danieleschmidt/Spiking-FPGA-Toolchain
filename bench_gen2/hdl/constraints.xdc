# Timing constraints for SNN implementation
# Clock period: 10.00 ns (100 MHz)

create_clock -period 10.00 -name clk [get_ports clk]

# Input/output delays
set_input_delay -clock clk 2.0 [all_inputs]
set_output_delay -clock clk 2.0 [all_outputs]

# Clock domain crossing constraints
set_false_path -from [get_ports rst]

# Area constraints for neuron placement
# create_pblock pblock_neurons
# resize_pblock pblock_neurons -add {SLICE_X0Y0:SLICE_X50Y50}
# add_cells_to_pblock pblock_neurons [get_cells -hier *neuron*]

# Memory interface timing
set_max_delay 10.0 -from [get_cells -hier *memory*] -to [get_cells -hier *axi*]