import math as m
from numpy.random import randint
from numpy.random import rand
import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
import time

class pvmodel:
    def __init__(self,Isc_stc,Voc_stc,Vmp_stc,Imp_stc,ki,kv,Ns,G_stc,T_stc,G,T):
        # constants
        tol = 6
        self.Voc_stc = Voc_stc
        self.tol = tol
        self.kv = kv
        k_boltz = 1.3807*10**(-23)
        q_elec  = 1.6022*10**(-19)

        # Thermal voltage of diodeB
        self.Vt      = (Ns*(k_boltz*(T+273)))/q_elec
        Vt_stc      = (Ns*(k_boltz*(T_stc+273)))/q_elec
        # Temp difference
        #deltaT = (T+273)-(T_stc+273)
        self.deltaT = T-T_stc

        # Short circuit current
        self.Isc = (Isc_stc+ki*self.deltaT)*(G/G_stc)

        # Open circuit voltage
        self.Voc = Voc_stc+kv*self.deltaT+self.Vt*m.log(G/G_stc)

        # MPP Current
        self.Imp = (Imp_stc+ki*self.deltaT)*(G/G_stc)

        # MPP voltage
        self.Vmp = Vmp_stc+(kv*self.deltaT)+self.Vt*m.log(G/G_stc)
        # Subject to
        self.Rs_max = ((self.Voc-self.Vmp)/(self.Imp))*2
        self.Rs_min = ((self.Voc-self.Vmp)/(self.Imp))/2
        self.Rp_max = (self.Vmp/(self.Isc-self.Imp))*2
        self.Rp_min = (self.Vmp/(self.Isc-self.Imp))/2
        self.n_max  = 2
        self.n_min  = 1
        self.n_diode = 1.5
        self.Rs_max_bin_length=len(bin(m.trunc(self.Rs_max*10**(tol)))[2:])
        self.Rp_max_bin_length=len(bin(m.trunc(self.Rp_max*10**(tol)))[2:])
        self.n_max_bin_length=len(bin(m.trunc(self.n_max*10**(tol)))[2:])
        print('Irradiance:',G,'W/(m^2)')
        print('Temperature:',T,'C')
        print('Voc STC:',Voc_stc,'V')
        print('Voc:',self.Voc,'V')
        print('Isc:',self.Isc,'A')
        print('Vmp:',self.Vmp,'V')
        print('Imp:',self.Imp,'A')

    def generate_solution(self):
        # Generate random solution
        Rs_guess = randint(m.trunc(self.Rs_min*10**(self.tol)), m.trunc(self.Rs_max*10**(self.tol)))/(10**(self.tol))
        Rp_guess = randint(m.trunc(self.Rp_min*10**(self.tol)), m.trunc(self.Rp_max*10**(self.tol)))/(10**(self.tol))
        n_guess = randint(m.trunc(self.n_min*10**(self.tol)), m.trunc(self.n_max*10**(self.tol)))/(10**(self.tol))
        solution = [Rs_guess, Rp_guess, n_guess]
        return solution

# to binary -------------------------------------

    def conv_bin(self,val):
        converted=bin(m.trunc(val*10**(self.tol)))
        return converted[2:]

#------------------------------------------------
# to int -------------------------------------

    def conv_int(self,string_val):
        string_val = '0b'+string_val
        converted=int(string_val,2)/(10**(self.tol))
        return converted

#------------------------------------------------
# mutation --------------------------------------

    def mutation(self,bitstring, r_mut):
        for i in range(len(bitstring)):
            if rand() < r_mut:
                bitstring=bitstring[:i] + str(1-int(bitstring[i])) + bitstring[i+1:]

        return bitstring

#------------------------------------------------


# crossover -------------------------------------
    def crossover(self,p1,p2,r_cross):
        p1_bin = [self.conv_bin(p1[0]).zfill(self.Rs_max_bin_length),self.conv_bin(p1[1]).zfill(self.Rp_max_bin_length),self.conv_bin(p1[2]).zfill(self.n_max_bin_length)]
        p2_bin = [self.conv_bin(p2[0]).zfill(self.Rs_max_bin_length),self.conv_bin(p2[1]).zfill(self.Rp_max_bin_length),self.conv_bin(p2[2]).zfill(self.n_max_bin_length)]
        c1, c2 = p1_bin.copy(), p2_bin.copy()
        if rand() < r_cross:
            # select random one of the three variables
            fit_val_c1 = float('inf')
            fit_val_c2 = float('inf')
            while(not(fit_val_c1 < float('inf')) or not(fit_val_c2 < float('inf'))):
                for var_cross in range(0,3):
                        # select crossover point that is not on the end of the string
                    point = randint(1, len(p1_bin[var_cross])-2)
                        # perform crossover
                    c1[var_cross] = p1_bin[var_cross][:point] + p2_bin[var_cross][point:]
                    c2[var_cross] = p2_bin[var_cross][:point] + p1_bin[var_cross][point:]
                fit_val_c1=self.fitness(self.conv_int(c1[0]),self.conv_int(c1[1]),self.conv_int(c1[2]))
                fit_val_c2=self.fitness(self.conv_int(c2[0]),self.conv_int(c2[1]),self.conv_int(c2[2]))
                #print(fit_val_c1,fit_val_c2)
        return [c1,c2]

#------------------------------------------------
# tournament selection --------------------------
    def tournament_selection(self,population, scores, k):
        selection_ix = randint(len(population))
        for ix in randint(0,len(population),m.ceil(len(population)*k)-1):
            if (scores[ix] < scores[selection_ix]):
                selection_ix = ix
        return population[selection_ix]
#------------------------------------------------

# create population -----------------------------
    def create_population(self,n_pop):
        population = list()
        population.clear()
        for i in range(n_pop):
            sol = self.generate_solution()
            #print(sol)
            population.append(sol)
        return population
# -----------------------------------------------

    # fitness calculation ---------------------------
    def fitness(self,Rs,Rp,n_diode):
        # Added last
        try:
            Gp = 1/Rp
        except ZeroDivisionError:
            Gp = float('inf')
        try:    
            Gamma = 1/(n_diode*self.Vt)
        except ZeroDivisionError:
            Gamma = float('inf')
        #Io = Isc/(m.exp((Voc+kv*deltaT)/n_diode/Vt)-1)
        try:
            Io = (self.Isc-(self.Voc-(self.Isc*Rs))/(Rp))*m.exp(-1*(self.Voc)/(n_diode*self.Vt))
        except ZeroDivisionError:
            Io = 0
        try:
            exponent = m.exp(Gamma*(self.Vmp+self.Imp*Rs))
        except OverflowError:
            exponent = float('inf')

        fitness3 = np.longdouble((Io*Gamma*exponent-Gp)/(1+Io*Gamma*Rs*exponent-Gp*Rs))
        #
        try:
            Iph = Io*m.exp((self.Voc)/(n_diode*self.Vt))+(self.Voc/Rp)
        except OverflowError:
            Iph = float('inf')
        except ZeroDivisionError:
            Iph = 0

        try:
            subeq = ((self.Isc*Rp-self.Voc+self.Isc*Rs)/(n_diode*self.Vt))*m.exp((self.Vmp+self.Imp*Rs-self.Voc)/(n_diode*self.Vt))+1
        except OverflowError:
            subeq = float('inf')
        except ZeroDivisionError:
            subeq = 0
        try:
            fitness1 = self.Imp-self.Vmp*(((1/Rp)*subeq)/(1+(Rs/Rp)*subeq))
        except ZeroDivisionError:
            fitness1 = float('inf')
        try:
            fitness2 = Iph-Io*m.exp((self.Vmp+self.Imp*Rs)/(n_diode*self.Vt))-((self.Vmp+self.Imp*Rs)/Rp)-self.Imp
        except OverflowError:
            fitness2 = float('inf')
        except ZeroDivisionError:
            fitness2 = float('inf')
        try:
            #Error = (fitness1**2)+(fitness2**2)
            Error = fitness2**2+fitness1**2
        except OverflowError:
            Error = float('inf')
        if(Rs > self.Rs_max):
            Error = float('inf')
        if(Rs < self.Rs_min):
            Error = float('inf')
        if(Rp > self.Rp_max):
            Error = float('inf')
        if(Rp < self.Rp_min):
            Error = float('inf')
        if(n_diode > self.n_max):
            Error = float('inf')
        if(n_diode < self.n_min):
            Error = float('inf')
        return Error
# -----------------------------------------------

# Genetic Algorithm

    def genetic_algorithm(self,n_pop,n_iter,r_cross,r_mut,k_select,min_conv):
        loop_count = 0
        population = self.create_population(n_pop)
        best, best_eval = list(), self.fitness(population[0][0],population[0][1],population[0][2])

        scores = list()
        pop_fit = list()
        #population = np.asarray(population)
        while (abs(best_eval) > min_conv):
            loop_count = loop_count + 1
            #print('Population',loop_count,population)
            scores.clear()
            for c in range(len(population)):
                scores.append(self.fitness(population[c][0],population[c][1],population[c][2]))
            
            best_fit = 1/min(scores)
            #print('Best fitness',best_fit)
            pop_fit.append(best_fit)
            for i in range(n_pop):
                #print(i)
                if scores[i] < best_eval:
                    best, best_eval = population[i],scores[i]
            #print('Best Fitness: ',best_eval,', Best Solution: ',best)
            selected = list()
            selected.clear()
            for i in range(len(population)):
                selected.append(self.tournament_selection(population, scores, k_select))
            children = list()
            children.clear()
            for k in range(0,n_pop,2):
                p1,p2=selected[k],selected[k+1]
                # iterate over two children
                for c in self.crossover(p1,p2,r_cross):
                    fit_val_c_mut = float('inf')
                    while(not(fit_val_c_mut < float('inf'))):
                        c_og = c
                        for i in range(0,3):
                            c[i] = self.mutation(c[i],r_mut)
                        fit_val_c_mut=self.fitness(self.conv_int(c[0]),self.conv_int(c[1]),self.conv_int(c[2]))
                        if(not(fit_val_c_mut < float('inf'))):
                            c=c_og
                    children.append([self.conv_int(c[0]),self.conv_int(c[1]),self.conv_int(c[2])])
            population=children
            if(loop_count >= n_iter):
                break
        return [self.Voc,self.Isc,self.Vmp,self.Imp,self.Vt,best, best_eval, loop_count,pop_fit]
# -----------------------------------------------
    def generate_model(self):
        begin = time.time()
        Voc,Isc,Vmp,Imp,Vt,best_sol, best_fitness, loops,pop_fit = self.genetic_algorithm(120,100,0.8,0.05,0.25,1*10**(-9))
        end = time.time()
        print(best_sol, best_fitness, loops,'; Time:',end-begin)
        Rs = best_sol[0]
        Rp = best_sol[1]
        n_diode = best_sol[2]
        Io = (Isc)/(m.exp(((self.Voc_stc+self.kv*self.deltaT)/n_diode)/Vt)-1)
        Iph = Isc

        def function_current(Vout,Iout):
            Ipv=Iph-Io*(m.exp((Vout+Iout*Rs)/(n_diode*Vt))-1)-(Vout+Iout*Rs)/Rp
            return Ipv-Iout
        # BISECTION METHOD FOR CONVERGENCE
        V_step = 0
        V_out_list = list()
        I_out_list = list()
        P_list     = list()
        tol_BS = 10**(-9)
        steps_needed = 500
        for V_step in np.linspace(0,Voc,m.ceil(steps_needed)):
            Vout = V_step
            n_max = 50
            n = 0
            a_c = 0
            b_c = Isc
            while n < n_max:
                p_c = (a_c+b_c)/2
                if(function_current(Vout,a_c)*function_current(Vout,p_c)<0):
                    b_c=p_c
                else:
                    if(function_current(Vout,b_c)*function_current(Vout,p_c)<0):
                        a_c=p_c
                    else:
                        break
                n = n + 1
            Iout = p_c
            V_out_list.append(Vout)
            I_out_list.append(Iout)
            P_list.append(Vout*Iout)

        del V_out_list[-1]
        del I_out_list[-1]
        V_out_list.append(Voc)
        I_out_list.append(0)
        """
        plt.figure(figsize = (8, 6))
        plt.plot(V_out_list, I_out_list, '-bo', label='Simulated Model')
        plt.legend()
        plt.xlabel('Voltage')
        plt.ylabel('Current')
        plt.plot(Vmp, Imp, "or")
        plt.plot(Voc, 0, "or")
        plt.plot(0, Isc, "or")
        plt.xticks(np.arange(0, Voc, 1.0))
        plt.yticks(np.arange(0, Isc+1, 0.5))
        plt.show()
        """
        """
        plt.figure(figsize = (8, 6))
        plt.plot(pop_fit, '-bo', label='Simulated Model')
        plt.show()
        """
        xs = V_out_list
        ys = I_out_list
        ps = list()

        for i in range(len(xs)):
            ps.append(xs[i]*ys[i])


        def function(x,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13):
            return c1 + c2 * x + c3 * x ** 2 + c4 * x ** 3 + c5 * x ** 4 + c6 * x ** 5 + c7 * x ** 6 + c8 * x ** 7 + c9 * x ** 8+ c10 * x ** 9+ c11 * x ** 10+ c12 * x ** 11+ c13 * x ** 12


        popt,cov = scipy.optimize.curve_fit(function, xs, ys)

        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13 = popt
        self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12,self.c13 = popt
        
        x_new_value = np.arange(min(xs), max(xs), 1)
        y_new_value = function(x_new_value,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13)
        p_fitted = x_new_value*y_new_value

        I_mp_fit = function(Vmp,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13)
        print(I_mp_fit)

        # Max power
        max_power_index_fitted = np.argmax(p_fitted)
        max_power_fit = p_fitted[max_power_index_fitted]
        print('Max power fit:',max_power_fit)
        max_power_voltage = x_new_value[max_power_index_fitted]
        max_power_current = y_new_value[max_power_index_fitted]
        max_power_actual = Vmp*Imp
        self.max_power_voltage = max_power_voltage
        self.max_power_current = max_power_current

        # Errors
        error_power = (abs(max_power_fit-max_power_actual)/max_power_actual)*100
        error_voltage = (abs(Vmp-max_power_voltage)/(Vmp))*100
        error_current = (abs(Imp-max_power_current)/(Imp))*100

        print('Error in maximum power: ',round(error_power,2),'%')
        print('Error in maximum voltage: ',round(error_voltage,2),'%')
        print('Error in maximum current: ',round(error_current,2),'%')

        """
        plt.plot(xs, ys, '-bo', label='Simulated Model')
        plt.plot(x_new_value,y_new_value,color="red")
        plt.plot(Vmp, Imp, "or")
        plt.plot(max_power_voltage, max_power_current, "og")
        plt.plot(Voc,0, "or")
        plt.plot(0,Isc, "or")
        plt.show()
        

        plt.plot(x_new_value,p_fitted)
        plt.plot(Vmp, Imp*Vmp, "or")
        plt.plot(x_new_value[max_power_index_fitted], p_fitted[max_power_index_fitted], "xg")
        plt.show()
        """
    def plot_fitted_iv(self):
        def function(x,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13):
            return c1 + c2 * x + c3 * x ** 2 + c4 * x ** 3 + c5 * x ** 4 + c6 * x ** 5 + c7 * x ** 6 + c8 * x ** 7 + c9 * x ** 8+ c10 * x ** 9+ c11 * x ** 10+ c12 * x ** 11+ c13 * x ** 12
        x_new_value = np.arange(0, self.Voc, 0.1)
        self.x_new_value = x_new_value
        y_new_value = function(x_new_value,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12,self.c13)
        plt.plot(x_new_value,y_new_value,color="blue")
        plt.xticks(np.arange(0, self.Voc, 1.0))
        plt.yticks(np.arange(0, self.Isc+1, 0.5))
        plt.plot(self.Vmp, self.Imp, "or")
        plt.plot(self.max_power_voltage, self.max_power_current, "xg")
        plt.grid()
        plt.show()

    def load_voltage_current(self,Rload):
        def function_load(x,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,Rload):
            return (c1 + c2 * x + c3 * x ** 2 + c4 * x ** 3 + c5 * x ** 4 + c6 * x ** 5 + c7 * x ** 6 + c8 * x ** 7 + c9 * x ** 8+ c10 * x ** 9+ c11 * x ** 10+ c12 * x ** 11+ c13 * x ** 12)- (1/Rload)*x

                #Load
        begin = time.time()
        n_max = 20
        n = 0
        a = 0
        b = self.Voc
        while n < n_max:
            p = (a+b)/2
            if(function_load(a,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12,self.c13,Rload)*function_load(p,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12,self.c13,Rload)<0):
                b=p
            else:
                if(function_load(b,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12,self.c13,Rload)*function_load(p,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12,self.c13,Rload)<0):
                    a=p
                else:
                    break
            n = n + 1
        end = time.time()
        print('Load convergence time:',end-begin)
        return p,(p/Rload)