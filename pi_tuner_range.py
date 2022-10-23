import math as m
from numpy.random import randint
from numpy.random import rand
import numpy as np
import control as ctr

# Initialize parameters
tol = 12
SimTime = 0.1
Ki_min = 0
Ki_max = 2
Kp_min = 0
Kp_max = 0.0001
T = 1e-03
Kd = 0

# Set binary maximum lengths for Ki and Kp
Kp_max_bin_length = len(bin(m.trunc(Kp_max * 10 ** (tol)))[2:])
Ki_max_bin_length = len(bin(m.trunc(Ki_max * 10 ** (tol)))[2:])


def generate_solution():

    # Generate random solution for Ki and Kp
    Kp_guess = randint(m.trunc(Kp_min * 10 ** (tol)),
                       m.trunc(Kp_max * 10 ** (tol)), 
                       dtype = np.int64) / (10 ** (tol))
    Ki_guess = randint(m.trunc(Ki_min * 10 ** (tol)), 
                       m.trunc(Ki_max * 10 ** (tol)),
                       dtype = np.int64) / (10 ** (tol))
    solution = [Kp_guess, Ki_guess]

    return solution

""" Convert decimal to binary """
def conv_bin(val):
    converted = bin(m.trunc(val * 10 ** (tol)))
    return converted[2:]

""" Convert vinary to decimal  """
def conv_int(string_val):
    string_val = '0b' + string_val
    converted = int(string_val, 2) / (10 ** (tol))

    return converted

""" Perform mutation step of genetic algorithm """
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring = (bitstring[:i] + str(1 - int(bitstring[i]))
                         + bitstring[i + 1:])

    return bitstring

""" Perform crossover step in genetic algorithm """
def crossover(Gz1, Gz2, p1, p2, r_cross):
    p1_bin = [conv_bin(p1[0]).zfill(Kp_max_bin_length), 
              conv_bin(p1[1]).zfill(Ki_max_bin_length)]
    p2_bin = [conv_bin(p2[0]).zfill(Kp_max_bin_length),
              conv_bin(p2[1]).zfill(Ki_max_bin_length)]
    c1, c2 = p1_bin.copy(), p2_bin.copy()

    if rand() < r_cross:
        fit_val_c1 = float('inf')
        fit_val_c2 = float('inf')

        while(not(fit_val_c1 < float('inf')) or not(fit_val_c2 < float('inf'))):
            for var_cross in range(0,2):
                
                # select crossover point that is not on the end of the string
                point = randint(1, len(p1_bin[var_cross]) - 2)
                
                # perform crossover
                c1[var_cross] = p1_bin[var_cross][:point] + p2_bin[var_cross][point:]
                c2[var_cross] = p2_bin[var_cross][:point] + p1_bin[var_cross][point:] 
            
            fit_val_c1 = fitness(Gz1, Gz2, conv_int(c1[0]), conv_int(c1[1]))
            fit_val_c2 = fitness(Gz1, Gz2, conv_int(c2[0]), conv_int(c2[1]))
    return [c1,c2]

""" Perform selection step in genetic algorithm """
def tournament_selection(population, scores, k):
    selection_ix = randint(len(population))
    for ix in randint(0, len(population), m.ceil(len(population) * k) - 1):
        if (scores[ix] < scores[selection_ix]):
            selection_ix = ix
    return population[selection_ix]

""" Create initial population of genetic algorithm """
def create_population(n_pop):
    population = list()
    population.clear()
    for i in range(n_pop):
        sol = generate_solution()
        population.append(sol)

    return population

# Calculate fitness of solution
def fitness(Gz1, Gz2, Kp, Ki):

    # Create transfer function of controller
    K_gain = Kp + Ki * T / 2 + Kd * 1 / T
    a = (Ki * T / 2 - Kp - 2 * Kd * 1 / T) / K_gain
    b = Kd / (T * K_gain)
    Dz = ctr.TransferFunction([K_gain, K_gain *a , b],[1, -1, 0], T)

    # Test the settling time and overshoot of the controller and calculate
    # the fitness
    try:
        sys1 = ctr.feedback(Dz * Gz1, 1)
        stepinfo1 = ctr.step_info(sys1, T = SimTime)
        sys2 = ctr.feedback(Dz * Gz2, 1)
        stepinfo2 = ctr.step_info(sys2, T = SimTime)
        if ((np.isnan(stepinfo1["SettlingTime"])) or 
            (np.isnan(stepinfo2["SettlingTime"]))):
            fitnessvalue = 999999
        elif ((Kp > Kp_max) or (Kp < Kp_min)  or (Ki > Ki_max)  or (Ki < Ki_min)
              or (stepinfo1["Overshoot"] > 0.1) or (stepinfo2["Overshoot"] > 0.1)):
            fitnessvalue = 999999
        else:
            fitnessvalue = ((stepinfo1["SettlingTime"] / 0.1) + (stepinfo2["SettlingTime"]
                            / 0.1) + (stepinfo1["Overshoot"] / 0.00001) 
                            + (stepinfo2["Overshoot"] / 0.00001))
    except:
        fitnessvalue = 999999

    return fitnessvalue

""" Main function of the genetic algorithm """
def genetic_algorithm(n_pop, n_iter, r_cross, 
                      r_mut, k_select, R_load1, Vload1, R_load2, Vload2):

    # Initialize the transfer function of the buck converter
    Vload = Vload1
    R_load = R_load1
    Iload = Vload/R_load
    resr = 0.1
    rd = 19.4e-03
    rsw = 14e-03
    rL = 16e-03
    L = 200e-06
    C = 136e-06
    Vf = 0.45
    Vs = 85
    K = Vload / Vs
    vx = Iload * (rd-rsw)
    rx = K * rsw + (1 - K) * rd

    # Numerator coefficients
    n_s_1 = R_load * resr * (Vf + Vs + vx) / (L * (R_load + resr))
    n_s_0 = R_load * (Vf + Vs + vx) / (C * L * (R_load + resr))
    num  = [n_s_1, n_s_0]

    # Denominator coefficients
    d_s_2 = 1
    d_s_1 = ((( 1 / C) *( 1 / (R_load + resr))) + ((1 / L) * 
             (rL + rx + (R_load * resr) / (R_load + resr))))
    d_s_0 = (R_load + rL + rx) / (C * L * (R_load + resr))
    den  = [d_s_2, d_s_1, d_s_0]

    # Construct final transfer function
    Gs_v1 = ctr.TransferFunction(num, den)

    # Repeat process for other side of range
    Vload = Vload2
    R_load = R_load2
    Iload = Vload/R_load
    resr = 0.1
    rd = 19.4e-03
    rsw = 14e-03
    rL = 16e-03
    L = 200e-06
    C = 136e-06
    Vf = 0.45
    Vs = 85
    K = Vload / Vs
    vx = Iload * (rd-rsw)
    rx = K * rsw + (1 - K) * rd

    # Numerator coefficients
    n_s_1 = R_load * resr * (Vf + Vs + vx) / (L * (R_load + resr))
    n_s_0 = R_load * (Vf + Vs + vx) / (C * L * (R_load + resr))
    num  = [n_s_1, n_s_0]

    # Denominator coefficients
    d_s_2 = 1
    d_s_1 = ((( 1 / C) *( 1 / (R_load + resr))) + ((1 / L) * 
             (rL + rx + (R_load * resr) / (R_load + resr))))
    d_s_0 = (R_load + rL + rx) / (C * L * (R_load + resr))
    den  = [d_s_2, d_s_1, d_s_0]
    
    # Construct final transfer function
    Gs_v2 = ctr.TransferFunction(num, den)

    # Sample both systems
    Gz1 = ctr.sample_system(Gs_v1, T)
    Gz2 = ctr.sample_system(Gs_v2, T)

    # Intialize genetic algorithm
    loop_count = 0
    population = create_population(n_pop)
    best, best_eval = list(), fitness(Gz1,Gz2,population[0][0],population[0][1])
    scores = list()
    pop_fit = list()
    best_eval = 99999999999

    # Start genetic algorithm
    while (loop_count < n_iter):

        loop_count = loop_count + 1
        scores.clear()

        for c in range(len(population)):
            scores.append(fitness(Gz1, Gz2, population[c][0], population[c][1]))
            
        best_fit = 1 / min(scores)

        pop_fit.append(best_fit)
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = population[i],scores[i]

        selected = list()
        selected.clear()
        for i in range(len(population)):
            selected.append(tournament_selection(population, scores, k_select))

        children = list()
        children.clear()

        for k in range(0, n_pop, 2):
            p1, p2 = selected[k], selected[k+1]
            for c in crossover(Gz1, Gz2, p1, p2, r_cross):
                fit_val_c_mut = float('inf')
                while(not(fit_val_c_mut < float('inf'))):
                    c_og = c
                    for i in range(0, 2):
                        c[i] = mutation(c[i], r_mut)

                    fit_val_c_mut = fitness(Gz1, Gz2, conv_int(c[0]),
                                            conv_int(c[1]))
                    if(not(fit_val_c_mut < float('inf'))):
                        c = c_og
                children.append([conv_int(c[0]), conv_int(c[1])])

        population = children
        if(loop_count >= n_iter):
            break

        Kp = best[0]
        Ki = best[1]
        K_gain = Kp + Ki * T / 2 + Kd * 1 / T
        a = (Ki * T / 2 - Kp - 2 * Kd * 1 / T) / K_gain
        b = Kd / (T * K_gain)

        try:
            Dz = ctr.TransferFunction([K_gain, K_gain * a, b], [1, -1, 0], T)
            sys = ctr.feedback(Dz * Gz1, 1)
            stepinfo1 = ctr.step_info(sys, T=SimTime)
            Dz = ctr.TransferFunction([K_gain, K_gain * a, b], [1, -1, 0], T)
            sys = ctr.feedback(Dz * Gz2, 1)
            stepinfo2 = ctr.step_info(sys, T=SimTime)
        except:
            print('fail')

    return best_fit, best, stepinfo1["Overshoot"], stepinfo1["SettlingTime"], stepinfo2["Overshoot"], stepinfo2["SettlingTime"]


Load_Voltage = [4.25, 4.25, 4.25, 10, 10, 20, 30, 20, 40, 10, 30, 50, 40, 
                60, 50, 70, 60, 20, 70, 30, 40, 50, 60, 70]
Load_Resistance = [0.2125, 0.289, 0.452, 0.5, 0.833, 1, 1.5, 1.82, 2, 2.5, 
                   2.8125, 3.125, 3.81, 4.5, 5.95, 6.125, 8.57, 10, 11.67, 
                   22.5, 40, 62.5, 90, 122.5]

solutions = list()

for i in range(len(Load_Voltage) - 1):
    best_fit, sol, Po1, Ts1, Po2, Ts2 = genetic_algorithm(100, 10, 0.85, 0.05,
                                                          0.25, Load_Resistance[i],
                                                          Load_Voltage[i], 
                                                          Load_Resistance[i + 1],
                                                          Load_Voltage[i + 1])
    oppoint_sol = [sol[0], sol[1], Load_Resistance[i], Load_Resistance[i + 1], Po1,
                   Ts1, Po2, Ts2]
    solutions.append(oppoint_sol)
    print(oppoint_sol, ",")

Rlower = solutions[0][2]
Rupper = solutions[0][3]
print(Rlower)
print(Rupper)

