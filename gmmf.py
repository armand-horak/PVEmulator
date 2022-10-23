import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Solutions for the PI controller
solutions = [[6.6303698e-05, 4.004931743009, 0.2125, 0.289, 0, 0.0095, 
              0, 0.01] ,
            [7.6846845e-05, 4.975782210194, 0.289, 0.452, 0, 0.00725, 0, 0.008],
            [5.4439448e-05, 7.012745665083, 0.452, 0.5, 0, 0.00475, 0, 0.005],
            [4.9928883e-05, 7.63994707475, 0.5, 0.833, 7.771561172376093e-13, 
             0.00425, 0, 0.005],
            [6.900165e-05, 9.995631691383, 0.833, 1, 0, 0.0035, 0, 0.0035],
            [1.349203e-05, 9.9242339175, 1, 1.5, 0, 0.0035, 0, 0.00375],
            [8.629537e-06, 9.893987257985, 1.5, 1.82, 0, 0.00375, 0, 0.004],
            [8.8369713e-05, 9.9369427136, 1.82, 2, 0, 0.004, 0, 0.004],
            [7.9801615e-05, 9.857879251605, 2, 2.5, 0, 0.004, 0, 0.004],
            [5.0631029e-05, 9.962381038977, 2.5, 2.8125, 0, 0.004, 0, 0.004],
            [1.1492114e-05, 9.922894695606, 2.8125, 3.125, 0, 0.004, 0, 0.004],
            [7.2354159e-05, 9.894451629099, 3.125, 3.81, 0, 0.004, 0, 0.004],
            [4.6150203e-05, 9.746415097572, 3.81, 4.5, 0, 0.00425, 
             1.665567683772906e-08, 0.004],
            [7.90273e-07, 8.177798386682, 4.5, 5.95, 0, 0.005,
             5.773159728050811e-13, 0.005],
            [1.37994e-07, 8.143352553284, 5.95, 6.125, 4.440892098500626e-14,
             0.005, 3.7898573168604333e-10, 0.005],
            [1.3455e-08, 6.975097315405, 6.125, 8.57, 0, 0.006,
             1.085063150441101e-08, 0.006],
            [1.179e-09, 6.567026638771, 8.57, 10, 0, 0.00625,
             1.694635543003642e-08, 0.00625],
            [9.09e-09, 6.065031397501, 10, 11.67, 4.440892098500624e-14, 0.007,
             1.9424806207979376e-08, 0.007],
            [5.82295e-07, 4.827594657511, 11.67, 22.5, 0, 0.009000000000000001,
             1.0496270519411148e-08, 0.009000000000000001],
            [3.76797e-07, 4.16622227025, 22.5, 40, 0, 0.0105, 4.196487601859638e-09,
             0.01025],
            [2.9748e-08, 3.823601291942, 40, 62.5, 0, 0.0115, 1.006794647651076e-09,
             0.01125],
            [5.3153e-08, 3.703354455781, 62.5, 90, 6.661338147750934e-14, 0.012,
             1.4802248315959335e-08, 0.012],
            [2.23718e-07, 3.556286805602, 90, 122.5, 1.2212453270876722e-12, 0.01225,
             3.259370551234035e-09, 0.01225]]

# Graph font
font = {'family' : 'serif', 'size':16}
matplotlib.rc('font', **font)

# Initialize variables
sol_list = list()
solutions = np.asarray(solutions)
counter = 0

# Generate fuzzy model for various resistances
for resistance in range(2, 2000, 2):
    resistance = resistance / 20
    gmmf_list = list()
    Kp_sum = 0 
    Ki_sum = 0
    weight_sum = 0

    # Find solution of fuzzy model
    for i in range(len(solutions)):
        Kp = solutions[i][0]
        Ki = solutions[i][1]
        Rlow = solutions[i][2]
        Rhigh = solutions[i][3]
        Rwidth = Rhigh-Rlow
        Rcenter = (Rhigh+Rlow)/2
        try:
            gmmf = np.exp(( - (resistance - Rcenter) ** 2) / (2 * Rwidth ** 2))
        except:
            gmmf = 0
        if(gmmf < 1e-6):
            gmmf = 0
        Kp_sum = Kp_sum + Kp * gmmf
        Ki_sum = Ki_sum + Ki * gmmf
        weight_sum = weight_sum + gmmf

    print("{", resistance, ",", Kp_sum / weight_sum, ",", Ki_sum / weight_sum,
          ",", 0, " }", ",")

    counter += 1

# Create plot of fuzzy model
x = np.arange(0.1, 100, 0.1)

for i in range(len(solutions)):
    Rlow = solutions[i][2]
    Rhigh = solutions[i][3]
    Rwidth = Rhigh-Rlow
    Rcenter = (Rhigh + Rlow) / 2
    try:
        gmmf = np.exp((-(x - Rcenter) ** 2) / (2 * Rwidth ** 2))
    except:
        gmmf = 0
    plt.plot(x,gmmf)

plt.title('Fuzzy model', color='#3d3d3d')
plt.xlabel('Load Resistance [Ω]', color='#3d3d3d')
plt.ylabel('Membership Value', color='#3d3d3d')
plt.grid()
plt.show()