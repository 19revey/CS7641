import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

from mlrose_hiive import MaxKColorGenerator, TSPGenerator
from mlrose_hiive import SARunner, GARunner, NNGSRunner, GeomDecay,RHCRunner,OneMax,FourPeaks,MIMICRunner
import mlrose_hiive as mlrose


class runner:
    def __init__(self,problem,algorithm) -> None:
        self.problem = problem
        self.algorithm = algorithm
        if algorithm == 'SA':
            self.runner = SARunner(problem=self.problem,
                                   experiment_name='SA',
                                   seed=1,
                                   iteration_list=2 ** np.arange(12),
                                   max_attempts=50,
                                   max_iters=1000,
                                   temperature_list=[100],
                                   decay_list=[GeomDecay])
        elif algorithm == 'GA':
            self.runner = GARunner(problem=self.problem,
                                    experiment_name='GA',
                                    seed=1,
                                    iteration_list=2 ** np.arange(12),
                                    max_attempts=50,
                                    population_sizes=[10],
                                    mutation_rates=[0.1])
        elif algorithm == 'RHC':
            self.runner = RHCRunner(problem=problem,
                            experiment_name='RHC',
                            seed=19,
                            iteration_list=2 ** np.arange(12),
                            max_attempts=50,
                            restart_list=[10])   
        elif algorithm == 'MIMIC':
            self.runner = MIMICRunner(problem=problem,
                            experiment_name='MIMIC',
                            seed=19,
                            iteration_list=2 ** np.arange(12),
                            max_attempts=50,
                            population_sizes=[10],
                            keep_percent_list=[0.1]
                            )   
    def run(self):
        df_run_stats, df_run_curves= self.runner.run()
        return df_run_stats,df_run_curves
    
    def best_hyperparameters(self):

        if self.algorithm == 'SA':
            df_run_stats,df_run_curves=self.run()
            best_fitness = df_run_curves['Fitness'].max()
            best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
            minimum_evaluations = best_runs['FEvals'].max()
            best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
            best_temp = best_curve_run['Temperature'].iloc()[0]
            best_max_iter = best_curve_run['max_iters'].iloc()[0]
            best_time = best_curve_run['Time'].iloc()[0]
            print(f'Best temperature: {best_temp}, best max iteration: {best_max_iter},best time: {best_time}')
            return best_fitness

        if self.algorithm == 'GA':
            df_run_stats,df_run_curves=self.run()
            best_fitness = df_run_curves['Fitness'].max()
            best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
            minimum_evaluations = best_runs['FEvals'].max()
            best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
            best_mr = best_curve_run['Mutation Rate'].iloc()[0]
            best_pop_size = best_curve_run['Population Size'].iloc()[0]
            best_time = best_curve_run['Time'].iloc()[0]
            print(f'Best Mutation Rate: {best_mr}, best Population Size: {best_pop_size}')
            return best_fitness
        
        if self.algorithm == 'RHC':
            df_run_stats,df_run_curves=self.run()
            best_fitness = df_run_curves['Fitness'].max()
            best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
            minimum_evaluations = best_runs['FEvals'].max()
            best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
            best_mr = best_curve_run['Restarts'].iloc()[0]
            best_time = best_curve_run['Time'].iloc()[0]
            print(f'Best Restarts: {best_mr}, best time: {best_time}')
            return best_fitness

    def plot_run_curves(self,column):
        df_run_stats,df_run_curves=self.run()

        fig = plt.figure(figsize=(4,3))

        if column == 'Temperature':
            df_run_curves['Temperature'] = df_run_curves['Temperature'].astype(str)

        for i in df_run_curves[column].unique():
            mutation_rate = i
            temp = df_run_curves[df_run_curves[column] == mutation_rate]
            temp.reset_index(inplace=True)
            plt.plot(temp['Time'], temp['Fitness'], label=f'{column}: ' + str(mutation_rate))
       
       
       
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Fitness vs Iteration')
        plt.legend(loc='best')
        
        plt.rcParams.update({
            'font.size': 10,          # General font size
            'axes.titlesize': 8,     # Font size for titles
            'axes.labelsize': 8,     # Font size for x and y labels
            # 'xtick.labelsize': 8,    # Font size for x-axis tick labels
            # 'ytick.labelsize': 8,    # Font size for y-axis tick labels
            'legend.fontsize': 8     # Font size for legend
        })

        plt.show()
