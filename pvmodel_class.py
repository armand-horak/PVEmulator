import math as m
import numpy as np
from numpy.random import randint,rand
from scipy.special import lambertw
import time

class pvmodel:

    """ Initialise PV Model """
    def __init__(self, Isc_stc, Voc_stc, Vmp_stc, Imp_stc,
        ki, kv, N_series, N_strings, G_stc, T_stc, G, T ):
        
        # Calculate Thermal Voltage:
        k_boltz = 1.3807 * 10**(-23)
        q_elec  = 1.6022 * 10**(-19)
        self.Vt = (k_boltz * (T + 273)) / q_elec

        # Calculate Temperature Difference in degrees Celcius:
        self.deltaT = T-T_stc

        # Voc and Isc temperature coefficients:
        kv = kv/100
        ki = ki/100
        self.kv = kv
        self.ki = ki
        
        # Adjust Voc for T and G:
        self.Voc = (Voc_stc + Voc_stc * kv * self.deltaT
                 + N_series * self.Vt * m.log(G / G_stc))

        # Adjust Vmp for T and G:
        self.Vmp = (Vmp_stc + Vmp_stc * kv * self.deltaT
                 + N_series * self.Vt * m.log(G / G_stc))

        # Adjust Imp for T and G:
        self.Imp = (Imp_stc + Imp_stc * ki * self.deltaT) * (G / G_stc)

        # Adjust Isc for T and G:
        self.Isc = (Isc_stc + Isc_stc * ki * self.deltaT) * (G / G_stc)
                    
        # Set Number Of Decimal Places For Solution:
        tol = 6
        self.tol = tol

        # Raise Exceptions For Warnings:
        np.seterr('raise')
                
        # Cell Arrangement:
        Ns = N_series * N_strings

        # Set Search Boundaries of Genetic Algorithm:
        self.Rs_max = ((self.Voc - self.Vmp) / (self.Imp)) 
        self.Rs_min = ((self.Voc - self.Vmp) / (self.Imp)) / 1000
        self.Rp_max = (self.Vmp / (self.Isc - self.Imp)) * 1000
        self.Rp_min = (self.Vmp / (self.Isc - self.Imp))  
        self.Ns_max = Ns
        self.Ns_min = Ns-1
        self.n_max  = 3
        self.n_min  = 0
        self.n_diode = 2

        # Set Maximum Binary Length For Parameters Used In Genetic Algorithm:
        self.Rs_max_bin_length = len(bin(m.trunc(self.Rs_max * 10**(tol)))[2:])
        self.Rp_max_bin_length = len(bin(m.trunc(self.Rp_max * 10**(tol)))[2:])
        self.n_max_bin_length = len(bin(m.trunc(self.n_max * 10**(tol)))[2:])
        self.Ns_max_bin_length = len(bin(m.trunc(self.Ns_max * 10**(tol)))[2:])

    """ Create Initial Population """
    def create_population(self,n_pop):
        population = list()
        population.clear()
        for i in range(n_pop):
            sol = self.generate_solution()
            population.append(sol)
        return population

    """ Generate Random Solutions For Each Parameter  """
    def generate_solution(self):   
        Rs_guess = randint(
                 m.trunc(self.Rs_min * 10**(self.tol)), 
                 m.trunc(self.Rs_max * 10**(self.tol))) / (10**(self.tol))
        Rp_guess = randint(
                 m.trunc(self.Rp_min * 10**(self.tol)),
                 m.trunc(self.Rp_max * 10**(self.tol)), 
                 dtype=np.int64) / (10**(self.tol))
        n_guess = randint(
                m.trunc(self.n_min * 10**(self.tol)),
                m.trunc(self.n_max * 10**(self.tol))) / (10**(self.tol))
        Ns_guess = randint(
                 m.trunc(self.Ns_min * 10**(self.tol)), 
                 m.trunc(self.Ns_max * 10**(self.tol))) / (10**(self.tol))
        
        return [Rs_guess, Rp_guess, n_guess, Ns_guess]


    """ Perform Mutation Step in Genetic Algorithm """
    def mutation(self, bitstring, r_mut): 
        for i in range(len(bitstring)):
            if rand() < r_mut:
                bitstring = (bitstring[:i]
                          + str(1-int(bitstring[i])) + bitstring[i+1:])
        return bitstring

    """ Perform Crossover Step in Genetic Algorithm """
    def crossover(self, p1, p2, r_cross):

        # Convert Both Parents To Binary:
        p1_bin = [self.conv_bin(p1[0]).zfill(self.Rs_max_bin_length), 
               self.conv_bin(p1[1]).zfill(self.Rp_max_bin_length), 
               self.conv_bin(p1[2]).zfill(self.n_max_bin_length), 
               self.conv_bin(p1[3]).zfill(self.Ns_max_bin_length)]

        p2_bin = [self.conv_bin(p2[0]).zfill(self.Rs_max_bin_length), 
               self.conv_bin(p2[1]).zfill(self.Rp_max_bin_length),
               self.conv_bin(p2[2]).zfill(self.n_max_bin_length),
               self.conv_bin(p2[3]).zfill(self.Ns_max_bin_length)]

        # Children Are By Definition A Copy Of Their Parents:
        c1, c2 = p1_bin.copy(), p2_bin.copy()
        
        # If Children Are Selected For Crossover:
        if rand() < r_cross:

            # Intialise Children Fitness To Minimum:
            fit_val_c1 = float('inf')
            fit_val_c2 = float('inf')
            
            while(not(fit_val_c1 < float('inf'))
                  or not(fit_val_c2 < float('inf'))):
                for var_cross in range(0,4):

                    # Select Crossover Point That Is Not At The End Of The String:
                    point = randint(1, len(p1_bin[var_cross]) - 2)
                    
                    # Perform Crossover:
                    c1[var_cross] = (p1_bin[var_cross][:point]
                                  + p2_bin[var_cross][point:])

                    c2[var_cross] = (p2_bin[var_cross][:point]
                                  + p1_bin[var_cross][point:])

                # Calculate Fitness Of Children:                  
                fit_val_c1 = self.fitness(self.conv_int(c1[0]),
                           self.conv_int(c1[1]), self.conv_int(c1[2]),
                           self.conv_int(c1[3]))

                fit_val_c2 = self.fitness(self.conv_int(c2[0]),
                           self.conv_int(c2[1]), self.conv_int(c2[2]),
                           self.conv_int(c2[3]))

        return [c1,c2]


    """ Perform Tournament Selection For Genetic Algorithm """
    def tournament_selection(self, population, scores, k):
        selection_ix = randint(len(population))
        for ix in randint(0, len(population), m.ceil(len(population) * k) - 1):
            if (scores[ix] < scores[selection_ix]):
                selection_ix = ix
        return population[selection_ix]


    """ Calculate Fitness Of Solution For Genetic Algorithm"""
    def fitness(self, Rs, Rp, n_diode, Ns):
        
        fitness1 = float('inf')
        fitness2 = float('inf')

        # Calculate Parameters:
        try:
            Io = ((self.Isc - (self.Voc - (self.Isc * Rs)) / Rp)
               * m.exp(-1 * (self.Voc) / (n_diode * Ns * self.Vt)))
        except:
            Io = float('inf')

        try:
            Iph = (Io * m.exp((self.Voc)
                / (n_diode * Ns * self.Vt))+ (self.Voc / Rp))
        except:
            Iph = float('inf')

        try:
            subeq = (((self.Isc * Rp - self.Voc + self.Isc * Rs)
                  / (n_diode * Ns * self.Vt)) * m.exp((self.Vmp
                  + self.Imp * Rs - self.Voc)
                  / (n_diode * Ns * self.Vt)) + 1)
        except:
            subeq = float('inf')

        # Calculate Fitness Function Values:
        try:
            fitness1 = (self.Imp - self.Vmp
                     * (((1 / Rp) * subeq) / (1 + (Rs / Rp) * subeq)))
        except:
            fitness1 = float('inf')

        try:
            fitness2 = (Iph - Io * m.exp((self.Vmp + self.Imp * Rs)
                     / (n_diode * Ns * self.Vt)) - ((self.Vmp
                     + self.Imp * Rs) / Rp) - self.Imp)
        except:
            fitness2 = float('inf')

        # Calculate Total Error For Genetic Algorithm:
        try:
            Error = (fitness1 ** 2) + (fitness2 ** 2)
        except:
            Error = float('inf')
        
        # If Any Parameter Is Out Of Bounds Maximise Its Error:
        if((Rs > self.Rs_max) or (Rs < self.Rs_min)
          or (Rp > self.Rp_max) or (Rp < self.Rp_min)
          or (n_diode > self.n_max) or (n_diode < self.n_min)
          or (Ns > self.Ns_max) or (Ns < self.Ns_min)):
            Error = float('inf')

        return Error


    """ Convert Int To Binary """
    def conv_bin(self, val):
        return bin(m.trunc(val * 10**(self.tol)))[2:]


    """ Convert Binary To Int """
    def conv_int(self, string_val):
        return int('0b'+string_val, 2) / (10**(self.tol))


    """ Genetic Algorithm Main Function """
    def genetic_algorithm(self, n_pop, n_iter, r_cross,
        r_mut, k_select, min_conv):

        # Intialise Genetic Algorithm::
        loop_count = 0
        population = self.create_population(n_pop)
        best, best_eval = list(), self.fitness(population[0][0], 
                        population[0][1], population[0][2], population[0][3])
        scores = list()
        pop_fit = list()

        # Search For Solution Until Within Tolerance 
        # OR Maximum Iterations Has Been Reached:
        while((abs(best_eval) > min_conv) and loop_count < n_iter):

            # Initialise Loop:
            loop_count = loop_count + 1
            scores.clear()
            selected = list()
            selected.clear()
            children = list()
            children.clear()

            # Calculate Fitness Of Each Child:
            for c in range(len(population)):
                scores.append(self.fitness(population[c][0], 
                              population[c][1], population[c][2], 
                              population[c][3]))

            # Find The Best Fitness:
            best_fit = 1 / min(scores)
            pop_fit.append(best_fit)
            
            # Find The Solution Having The Best Fitness:
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = population[i], scores[i]

            # Select From The Population For Crossover:
            for i in range(len(population)):
                selected.append(self.tournament_selection(population,
                                scores, k_select))                

            # Select Two Parents Each Time And Go Through Population:
            for k in range(0, n_pop, 2):
                p1, p2 = selected[k], selected[k+1]
                
                # Perform Crossover To Create Children From Parents:
                for c in self.crossover(p1, p2, r_cross):
                    fit_val_c_mut = float('inf')

                    #  Mutate New Children:
                    while(not(fit_val_c_mut < float('inf'))):
                        c_og = c
                        for i in range(0, 3):
                            c[i] = self.mutation(c[i], r_mut)
                        
                        # Evalute Mutated Childrens' Fitness:    
                        fit_val_c_mut = self.fitness(self.conv_int(c[0]),
                                      self.conv_int(c[1]), self.conv_int(c[2]),
                                      self.conv_int(c[3]))
                        if(not(fit_val_c_mut < float('inf'))):
                            c=c_og

                    # Add The New Children If Their Fitness Is Defined:    
                    children.append([self.conv_int(c[0]), self.conv_int(c[1]),
                                    self.conv_int(c[2]), self.conv_int(c[3])])
            
            # The Next Population Is The Children:
            population = children

        return [self.Voc, self.Isc, self.Vmp, self.Imp, self.Vt,
                best, best_eval, loop_count, pop_fit]


    """ Generate The PV Model """
    def generate_model(self):

        # Start A Timer:
        begin = time.time()

        # Generate The Model Using The Genetic Algorithm:
        (Voc, Isc, Vmp, 
        Imp, Vt, best_sol, 
        best_fitness, 
        loops, pop_fit) = self.genetic_algorithm(100, 25, 0.8, 0.05, 0.4,
                                                 1 * 10**(-6))

        # Stop The Timer:
        end = time.time()

        # Display The Best Solution Along With The Elapsed Time:
        print(best_sol, best_fitness, loops, '; Time:', end-begin)
        
        # Extract The Solution Parameters
        Rs = best_sol[0]
        Rp = best_sol[1]
        n_diode = best_sol[2]
        Ns = best_sol[3]
        Io = ((self.Isc - (self.Voc - (self.Isc * Rs)) / Rp)
               * m.exp(-1 * (self.Voc) / (n_diode * Ns * self.Vt)))
        self.Rs = Rs
        self.Rp = Rp
        self.n_diode = n_diode
        self.Ns = Ns
        self.Io = Io

        """ Lambert W Function For Construction of I-V Curve """
        def lambert_w_current(V_lw):
            return (((Rp * (Isc + Io) - V_lw) / (Rs + Rp))
                      - (n_diode * Ns * self.Vt / Rs)
                      * lambertw(((Rs * Rp * Io)
                      / (n_diode * Ns * self.Vt * (Rs + Rp)))
                      * np.exp((Rp * Rs * (Isc + Io) + Rp * V_lw)
                      / (n_diode * Ns * self.Vt * (Rs + Rp))))).real
        
        # Generate I, V, and P Arrays With Lambert W:
        voltage_lw = np.concatenate((np.arange(0, Vmp, Vmp / 50), 
                     np.arange(Vmp, Voc, (Voc - Vmp) / 50)))
        voltage_lw = voltage_lw.tolist()
        current_lw = list()

        for i in range(len(voltage_lw)):
          current_lw.append(lambert_w_current(voltage_lw[i]))
        
        current_lw.append(0)
        voltage_lw.append(Voc)
        voltage_lw = np.asarray(voltage_lw)
        current_lw = np.asarray(current_lw)
        self.current_lw = current_lw
        self.voltage_lw = voltage_lw
        p_fitted = voltage_lw*current_lw

        # Max Power Calculations:
        max_power_index_fitted = np.argmax(p_fitted)
        max_power_fit = p_fitted[max_power_index_fitted]
        max_power_voltage = voltage_lw[max_power_index_fitted]
        max_power_current = current_lw[max_power_index_fitted]
        max_power_actual = Vmp * Imp
        self.max_power_voltage = max_power_voltage
        self.max_power_current = max_power_current

        # Error Calculations:
        error_power = (abs(max_power_fit - max_power_actual)
                       / max_power_actual) * 100
        error_voltage = (abs(Vmp - max_power_voltage) / (Vmp)) * 100
        error_current = (abs(Imp - max_power_current) / (Imp)) * 100

        # If Error Is Larger Than Maximum Allowable Error, Rerun Function:
        error_max = 0.1
        if((error_power > error_max) or (error_current > (error_max)) 
            or (error_voltage > (error_max))):
            self.generate_model()
        #self.operating_points()
        return (np.asarray([voltage_lw, current_lw]), Voc, Isc, Vmp, Imp)


    """ Calculate Operating Point For Specific Load """
    def operating_points(self):
        
        """ Lambert W Intersected With Load Line """    
        def load_current(x,R_load):
            return (((self.Rp * (self.Isc + self.Io)
            - x) / (self.Rs + self.Rp))
            - (self.n_diode * self.Ns * self.Vt / self.Rs)
            * lambertw(((self.Rs * self.Rp * self.Io)
            / (self.n_diode * self.Ns * self.Vt
            * (self.Rs + self.Rp))) * np.exp((self.Rp * self.Rs
            * (self.Isc + self.Io) + self.Rp * x)
            / (self.n_diode * self.Ns * self.Vt
            * (self.Rs + self.Rp))))).real - (1/R_load)*x

        voltages = list()
        Load_Resistance = np.arange(0.1,100,0.1)

        # Find the intersection point for each of the loads
        for i in Load_Resistance:
            Rload = i
        # Bisection Method For Root Finding:
            n_max = 1000
            n = 0
            a = 0
            b = self.Voc
            tol = 1*10**(-3)

            while n < n_max:
                p = (a + b) / 2

                if(np.sign(load_current(p, Rload)) == np.sign(load_current(a, Rload))):
                    a = p
                else:
                    b = p

                test = load_current(p, Rload)

                if(abs(test) < tol):
                    break
                n = n + 1
            print(Rload,p)
            voltages.append(p)

        return voltages

        
