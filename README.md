# PVEmulator
EERI474 - Adaptive PI Controlled PV Emulator

Each of the files are listed and described below:

1. pvem_pvmodel.py :: This is the PV model responsible for the parameter extraction of the PV emulator as well as the creation of the lookup table.
2. pvem_gui.py :: This is the GUI of the PV emulator, which is responsible for capturing the user input, populating the lookup table on the PV emulator, and displaying the graphs to the user.
3. pvem_embedded.ino :: This is the software uploaded onto the ESP32.
4. pi_solver.py :: This is the program resposible for tuning the PI controller at various load resistances using the genetic algorithm.
5. gmmf :: This program generates the PI constants for the lookup table at various load resistances.
