# -*- coding: uIF-8 -*-
"""
Created on Thu Dec 28 14:17:40 2023

@author: Admin
"""

import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import TabularCPD
from scipy.stats import beta
import numpy as np

'Alpha beta parameters for OF node'
alpha_OF_param=[6, 8, 7.5, 9, 10]
beta_OF_param=[2, 3, 4, 3, 7]
x=[0, 0.5, 1]

# Create a directed graph
G_OF = nx.DiGraph()

nodes_OF=['OF_01', 'OF_02', 'OF_03', 'OF_04', 'OF_05', 'OF']

G_OF.add_nodes_from(nodes_OF)

edges_OF=[ ('OF_01', 'OF'), ('OF_02', 'OF'), ('OF_03', 'OF'), ('OF_04', 'OF'), ('OF_05', 'OF')]
G_OF.add_edges_from(edges_OF)

beta_dist_OF_01 = beta(alpha_OF_param[0], beta_OF_param[0])
beta_dist_OF_02 = beta(alpha_OF_param[1], beta_OF_param[1])
beta_dist_OF_03 = beta(alpha_OF_param[2], beta_OF_param[2])
beta_dist_OF_04 = beta(alpha_OF_param[3], beta_OF_param[3])
beta_dist_OF_05 = beta(alpha_OF_param[4], beta_OF_param[4])

# Generate x values
x_values = np.linspace(0, 1, 1000)

# Create subplots
fig_OF, axs_OF = plt.subplots(len(alpha_OF_param), 1, figsize=(8, 2 * len(alpha_OF_param)))

# Plot each beta distribution
for i in range(len(alpha_OF_param)):
    beta_dist = beta(alpha_OF_param[i], beta_OF_param[i])
    y_values = beta_dist.pdf(x_values)
    
    axs_OF[i].plot(x_values, y_values, label=f'Beta({alpha_OF_param[i]}, {beta_OF_param[i]})')
    axs_OF[i].set_title(f'Beta Distribution of OF_{i+1}')
    axs_OF[i].legend()

# Adjust layout
plt.tight_layout()
plt.show()

'OF_01 parameters probability'
prob_OF_01_0 = beta_dist_OF_01.cdf(x[1]) - beta_dist_OF_01.cdf(x[0])
prob_OF_01_1 = beta_dist_OF_01.cdf(x[2]) - beta_dist_OF_01.cdf(x[1])

'OF_02 parameters probability'
prob_OF_02_0 = beta_dist_OF_02.cdf(x[1]) - beta_dist_OF_02.cdf(x[0])
prob_OF_02_1 = beta_dist_OF_02.cdf(x[2]) - beta_dist_OF_02.cdf(x[1])

'OF_03 parameters probability'
prob_OF_03_0 = beta_dist_OF_03.cdf(x[1]) - beta_dist_OF_03.cdf(x[0])
prob_OF_03_1 = beta_dist_OF_03.cdf(x[2]) - beta_dist_OF_03.cdf(x[1])

'OF_04 parameters probability'
prob_OF_04_0 = beta_dist_OF_04.cdf(x[1]) - beta_dist_OF_04.cdf(x[0])
prob_OF_04_1 = beta_dist_OF_04.cdf(x[2]) - beta_dist_OF_04.cdf(x[1])

'OF_05 parameters probability'
prob_OF_05_0 = beta_dist_OF_05.cdf(x[1]) - beta_dist_OF_05.cdf(x[0])
prob_OF_05_1 = beta_dist_OF_05.cdf(x[2]) - beta_dist_OF_05.cdf(x[1])

OFmodel = BayesianNetwork(edges_OF)

cpd_OF_01 = TabularCPD(variable='OF_01', variable_card=2, values=[[prob_OF_01_0], [prob_OF_01_1]])
cpd_OF_02 = TabularCPD(variable='OF_02', variable_card=2, values=[[prob_OF_02_0], [prob_OF_02_1]])
cpd_OF_03 = TabularCPD(variable='OF_03', variable_card=2, values=[[prob_OF_03_0], [prob_OF_03_1]])
cpd_OF_04 = TabularCPD(variable='OF_04', variable_card=2, values=[[prob_OF_04_0], [prob_OF_04_1]])
cpd_OF_05 = TabularCPD(variable='OF_05', variable_card=2, values=[[prob_OF_05_0], [prob_OF_05_1]])

OF_0_values=[ 1, prob_OF_01_0, prob_OF_02_0, (1-min(prob_OF_01_1, prob_OF_02_1)), 
           prob_OF_03_0, (1-min(prob_OF_01_1, prob_OF_03_1)), (1-min(prob_OF_02_1, prob_OF_03_1)), (1-min(prob_OF_01_1, prob_OF_02_1, prob_OF_03_1)),
           prob_OF_04_0, (1-min(prob_OF_04_1, prob_OF_01_1)), (1-min(prob_OF_04_1, prob_OF_02_1)), (1-min(prob_OF_04_1, prob_OF_02_1, prob_OF_01_1)),
           (1-min(prob_OF_04_1, prob_OF_03_1)), (1-min(prob_OF_04_1, prob_OF_03_1, prob_OF_01_1)), (1-min(prob_OF_04_1, prob_OF_03_1, prob_OF_02_1)), (1-min(prob_OF_04_1, prob_OF_03_1, prob_OF_02_1, prob_OF_01_1)),
           prob_OF_05_0, (1-min(prob_OF_05_1, prob_OF_01_1)), (1-min(prob_OF_05_1, prob_OF_02_1)), (1-min(prob_OF_05_1, prob_OF_01_1, prob_OF_02_1)), 
           (1-min(prob_OF_05_1, prob_OF_03_1)), (1-min(prob_OF_05_1, prob_OF_03_1, prob_OF_01_1)), (1-min(prob_OF_05_1, prob_OF_03_1, prob_OF_02_1)), (1-min(prob_OF_05_1, prob_OF_03_1, prob_OF_02_1, prob_OF_01_1)),
           (1-min(prob_OF_05_1, prob_OF_04_1)), (1-min(prob_OF_05_1, prob_OF_04_1, prob_OF_01_1)), (1-min(prob_OF_05_1, prob_OF_04_1, prob_OF_02_1)), (1-min(prob_OF_05_1, prob_OF_04_1, prob_OF_02_1, prob_OF_01_1)),
           (1-min(prob_OF_05_1, prob_OF_04_1, prob_OF_03_1)), (1-min(prob_OF_05_1, prob_OF_04_1, prob_OF_03_1, prob_OF_01_1)), (1-min(prob_OF_05_1, prob_OF_04_1, prob_OF_03_1, prob_OF_02_1)), 0]
                                                                                                    
OF_1_values=[0, prob_OF_01_1, prob_OF_02_1, (min(prob_OF_01_1, prob_OF_02_1)), 
           prob_OF_03_1, (min(prob_OF_01_1, prob_OF_03_1)), (min(prob_OF_02_1, prob_OF_03_1)), (min(prob_OF_01_1, prob_OF_02_1, prob_OF_03_1)),
           prob_OF_04_1, (min(prob_OF_04_1, prob_OF_01_1)), (min(prob_OF_04_1, prob_OF_02_1)), (min(prob_OF_04_1, prob_OF_02_1, prob_OF_01_1)),
           (min(prob_OF_04_1, prob_OF_03_1)), (min(prob_OF_04_1, prob_OF_03_1, prob_OF_01_1)), (min(prob_OF_04_1, prob_OF_03_1, prob_OF_02_1)), (min(prob_OF_04_1, prob_OF_03_1, prob_OF_02_1, prob_OF_01_1)),
           prob_OF_05_1, (min(prob_OF_05_1, prob_OF_01_1)), (min(prob_OF_05_1, prob_OF_02_1)), (min(prob_OF_05_1, prob_OF_01_1, prob_OF_02_1)), 
           (min(prob_OF_05_1, prob_OF_03_1)), (min(prob_OF_05_1, prob_OF_03_1, prob_OF_01_1)), (min(prob_OF_05_1, prob_OF_03_1, prob_OF_02_1)), (min(prob_OF_05_1, prob_OF_03_1, prob_OF_02_1, prob_OF_01_1)),
           (min(prob_OF_05_1, prob_OF_04_1)), (min(prob_OF_05_1, prob_OF_04_1, prob_OF_01_1)), (min(prob_OF_05_1, prob_OF_04_1, prob_OF_02_1)), (min(prob_OF_05_1, prob_OF_04_1, prob_OF_02_1, prob_OF_01_1)),
           (min(prob_OF_05_1, prob_OF_04_1, prob_OF_03_1)), (min(prob_OF_05_1, prob_OF_04_1, prob_OF_03_1, prob_OF_01_1)), (min(prob_OF_05_1, prob_OF_04_1, prob_OF_03_1, prob_OF_02_1)), 1]

cpd_OF = TabularCPD(variable='OF', variable_card=2, values=[OF_0_values, OF_1_values], evidence=['OF_01', 'OF_02', 'OF_03', 'OF_04', 'OF_05'], evidence_card=[2, 2, 2, 2, 2])

pos_OF = nx.spring_layout(G_OF)
nx.draw(G_OF, pos_OF, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black", font_weight="bold")
plt.title("Directed Graph")
plt.show()

OFmodel.add_cpds(cpd_OF_01, cpd_OF_02, cpd_OF_03, cpd_OF_04, cpd_OF_05, cpd_OF)
OFmodel.get_cpds()

inference_OF = VariableElimination(OFmodel)

query_result_OF = inference_OF.query(variables=['OF'])
values_OF=query_result_OF.values
# Printing query results
prob_OF_0 = values_OF[0]
prob_OF_1 =  values_OF[1]
print('Probability of Organisational factor Error is ', prob_OF_0)

'''Supervisory factors'''
'Alpha beta parameters for SF node'
alpha_SF_param=[4.5, 5, 7, 9, 10, 5]
beta_SF_param=[1.8, 2, 3, 5, 5, 1.2]

# Create a directed graph
G_SF = nx.DiGraph()

nodes_SF=['SF_01', 'SF_02', 'SF_03', 'SF_04', 'SF_05', 'SF_06', 'OF', 'SF']

G_SF.add_nodes_from(nodes_SF)

edges_SF=[('SF_01', 'SF'), ('SF_02', 'SF'), ('SF_03', 'SF'), ('SF_04', 'SF'), ('SF_05', 'SF'), ('SF_06', 'SF'), ('OF', 'SF')]
G_SF.add_edges_from(edges_SF)

beta_dist_SF_01 = beta(alpha_SF_param[0], beta_SF_param[0])
beta_dist_SF_02 = beta(alpha_SF_param[1], beta_SF_param[1])
beta_dist_SF_03 = beta(alpha_SF_param[2], beta_SF_param[2])
beta_dist_SF_04 = beta(alpha_SF_param[3], beta_SF_param[3])
beta_dist_SF_05 = beta(alpha_SF_param[4], beta_SF_param[4])
beta_dist_SF_06 = beta(alpha_SF_param[5], beta_SF_param[5])

# Create subplots
fig_SF, axs_SF = plt.subplots(len(alpha_SF_param), 1, figsize=(8, 2 * len(alpha_SF_param)))

# Plot each beta distribution
for i in range(len(alpha_SF_param)):
    beta_dist = beta(alpha_SF_param[i], beta_SF_param[i])
    y_values = beta_dist.pdf(x_values)
    
    axs_SF[i].plot(x_values, y_values, label=f'Beta({alpha_SF_param[i]}, {beta_SF_param[i]})')
    axs_SF[i].set_title(f'Beta Distribution SF_{i+1}')
    axs_SF[i].legend()

# Adjust layout
plt.tight_layout()
plt.show()

'SF_01 parameters probability'
prob_SF_01_0 = beta_dist_SF_01.cdf(x[1]) - beta_dist_SF_01.cdf(x[0])
prob_SF_01_1 = beta_dist_SF_01.cdf(x[2]) - beta_dist_SF_01.cdf(x[1])

'SF_02 parameters probability'
prob_SF_02_0 = beta_dist_SF_02.cdf(x[1]) - beta_dist_SF_02.cdf(x[0])
prob_SF_02_1 = beta_dist_SF_02.cdf(x[2]) - beta_dist_SF_02.cdf(x[1])

'SF_03 parameters probability'
prob_SF_03_0 = beta_dist_SF_03.cdf(x[1]) - beta_dist_SF_03.cdf(x[0])
prob_SF_03_1 = beta_dist_SF_03.cdf(x[2]) - beta_dist_SF_03.cdf(x[1])

'SF_04 parameters probability'
prob_SF_04_0 = beta_dist_SF_04.cdf(x[1]) - beta_dist_SF_04.cdf(x[0])
prob_SF_04_1 = beta_dist_SF_04.cdf(x[2]) - beta_dist_SF_04.cdf(x[1])

'SF_05 parameters probability'
prob_SF_05_0 = beta_dist_SF_05.cdf(x[1]) - beta_dist_SF_05.cdf(x[0])
prob_SF_05_1 = beta_dist_SF_05.cdf(x[2]) - beta_dist_SF_05.cdf(x[1])

'SF_06 parameters probability'
prob_SF_06_0 = beta_dist_SF_06.cdf(x[1]) - beta_dist_SF_06.cdf(x[0])
prob_SF_06_1 = beta_dist_SF_06.cdf(x[2]) - beta_dist_SF_06.cdf(x[1])

SFmodel = BayesianNetwork(edges_SF)

cpd_SF_01 = TabularCPD(variable='SF_01', variable_card=2, values=[[prob_SF_01_0], [prob_SF_01_1]])
cpd_SF_02 = TabularCPD(variable='SF_02', variable_card=2, values=[[prob_SF_02_0], [prob_SF_02_1]])
cpd_SF_03 = TabularCPD(variable='SF_03', variable_card=2, values=[[prob_SF_03_0], [prob_SF_03_1]])
cpd_SF_04 = TabularCPD(variable='SF_04', variable_card=2, values=[[prob_SF_04_0], [prob_SF_04_1]])
cpd_SF_05 = TabularCPD(variable='SF_05', variable_card=2, values=[[prob_SF_05_0], [prob_SF_05_1]])
cpd_SF_06 = TabularCPD(variable='SF_06', variable_card=2, values=[[prob_SF_06_0], [prob_SF_06_1]])
cpd_OF_for_SF = TabularCPD(variable='OF', variable_card=2, values=[[prob_OF_0], [prob_OF_1]])

SF_0_values=[1, prob_SF_01_0, prob_SF_02_0, 1-min(prob_SF_01_1, prob_SF_02_1), 
             prob_SF_03_0, 1-min(prob_SF_03_1, prob_SF_01_1), 1-min(prob_SF_03_1, prob_SF_02_1), 1-min(prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             prob_SF_04_0, 1-min(prob_SF_04_1, prob_SF_01_1), 1-min(prob_SF_04_1, prob_SF_02_1), 1-min(prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 
             1-min(prob_SF_04_1, prob_SF_03_1), 1-min(prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1),
             prob_SF_05_0, 1-min(prob_SF_05_1, prob_SF_01_1), 1-min(prob_SF_05_1, prob_SF_02_1), 1-min(prob_SF_05_1, prob_SF_02_1, prob_SF_01_1), 
             1-min(prob_SF_05_1, prob_SF_03_1), 1-min(prob_SF_05_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_SF_05_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_SF_05_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1),
             1-min(prob_SF_05_1, prob_SF_04_1), 1-min(prob_SF_05_1, prob_SF_04_1, prob_SF_01_1), 1-min(prob_SF_05_1, prob_SF_04_1, prob_SF_02_1), 1-min(prob_SF_05_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1),
             1-min(prob_SF_05_1, prob_SF_04_1, prob_SF_03_1), 1-min(prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             prob_SF_06_0, 1-min(prob_SF_06_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_02_1), 1-min(prob_SF_06_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_03_1), 
             1-min(prob_SF_06_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_SF_06_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_04_1), 
             1-min(prob_SF_06_1, prob_SF_04_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_04_1, prob_SF_02_1), 1-min(prob_SF_06_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_04_1, prob_SF_03_1), 
             1-min(prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_05_1), 
             1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_02_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_03_1), 
             1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1),
             1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1), 
             1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             prob_OF_0, 1-min(prob_OF_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_03_1), 
             1-min(prob_OF_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_04_1), 
             1-min(prob_OF_1, prob_SF_04_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_04_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_04_1, prob_SF_03_1), 
             1-min(prob_OF_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_05_1), 
             1-min(prob_OF_1, prob_SF_05_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_03_1), 
             1-min(prob_OF_1, prob_SF_05_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_04_1), 
             1-min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1), 
             1-min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1), 
             1-min(prob_OF_1, prob_SF_06_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_02_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_03_1), 
             1-min(prob_OF_1, prob_SF_06_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1),  
             1-min(prob_OF_1, prob_SF_06_1, prob_SF_04_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 
             1-min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_03_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_02_1, prob_SF_01_1), 
             1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_03_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 
             1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), 1-min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 0]

                                                                                                    
SF_1_values=[0, prob_SF_01_1, prob_SF_02_1, min(prob_SF_01_1, prob_SF_02_1), 
             prob_SF_03_1, min(prob_SF_03_1, prob_SF_01_1), min(prob_SF_03_1, prob_SF_02_1), min(prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             prob_SF_04_1, min(prob_SF_04_1, prob_SF_01_1), min(prob_SF_04_1, prob_SF_02_1), min(prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 
             min(prob_SF_04_1, prob_SF_03_1), min(prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), min(prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), min(prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1),
             prob_SF_05_1, min(prob_SF_05_1, prob_SF_01_1), min(prob_SF_05_1, prob_SF_02_1), min(prob_SF_05_1, prob_SF_02_1, prob_SF_01_1), 
             min(prob_SF_05_1, prob_SF_03_1), min(prob_SF_05_1, prob_SF_03_1, prob_SF_01_1), min(prob_SF_05_1, prob_SF_03_1, prob_SF_02_1), min(prob_SF_05_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1),
             min(prob_SF_05_1, prob_SF_04_1), min(prob_SF_05_1, prob_SF_04_1, prob_SF_01_1), min(prob_SF_05_1, prob_SF_04_1, prob_SF_02_1), min(prob_SF_05_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1),
             min(prob_SF_05_1, prob_SF_04_1, prob_SF_03_1), min(prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), min(prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), min(prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             prob_SF_06_1, min(prob_SF_06_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_02_1), min(prob_SF_06_1, prob_SF_02_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_03_1), 
             min(prob_SF_06_1, prob_SF_03_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_03_1, prob_SF_02_1), min(prob_SF_06_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_04_1), 
             min(prob_SF_06_1, prob_SF_04_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_04_1, prob_SF_02_1), min(prob_SF_06_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_04_1, prob_SF_03_1), 
             min(prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), min(prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_05_1), 
             min(prob_SF_06_1, prob_SF_05_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_02_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_02_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_03_1), 
             min(prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1),
             min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1), 
             min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), min(prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             prob_OF_1, min(prob_OF_1, prob_SF_01_1), min(prob_OF_1, prob_SF_02_1), min(prob_OF_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_03_1), 
             min(prob_OF_1, prob_SF_03_1, prob_SF_01_1), min(prob_OF_1, prob_SF_03_1, prob_SF_02_1), min(prob_OF_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_04_1), 
             min(prob_OF_1, prob_SF_04_1, prob_SF_01_1), min(prob_OF_1, prob_SF_04_1, prob_SF_02_1), min(prob_OF_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_04_1, prob_SF_03_1), 
             min(prob_OF_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), min(prob_OF_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), min(prob_OF_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_05_1), 
             min(prob_OF_1, prob_SF_05_1, prob_SF_01_1), min(prob_OF_1, prob_SF_05_1, prob_SF_02_1), min(prob_OF_1, prob_SF_05_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_05_1, prob_SF_03_1), 
             min(prob_OF_1, prob_SF_05_1, prob_SF_03_1, prob_SF_01_1), min(prob_OF_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1), min(prob_OF_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_05_1, prob_SF_04_1), 
             min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_01_1), min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1), min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1), 
             min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), min(prob_OF_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1), 
             min(prob_OF_1, prob_SF_06_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_02_1), min(prob_OF_1, prob_SF_06_1, prob_SF_02_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_03_1), 
             min(prob_OF_1, prob_SF_06_1, prob_SF_03_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_03_1, prob_SF_02_1), min(prob_OF_1, prob_SF_06_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1),  
             min(prob_OF_1, prob_SF_06_1, prob_SF_04_1), min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_02_1), min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 
             min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_03_1), min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), min(prob_OF_1, prob_SF_06_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             min(prob_OF_1, prob_SF_06_1, prob_SF_05_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_02_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_02_1, prob_SF_01_1), 
             min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_03_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_03_1, prob_SF_02_1, prob_SF_01_1), 
             min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_02_1, prob_SF_01_1), 
             min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_01_1), min(prob_OF_1, prob_SF_06_1, prob_SF_05_1, prob_SF_04_1, prob_SF_03_1, prob_SF_02_1), 1]


cpd_SF = TabularCPD(variable='SF', variable_card=2, values=[SF_0_values, SF_1_values], evidence=['SF_01', 'SF_02', 'SF_03', 'SF_04', 'SF_05', 'SF_06', 'OF'], evidence_card=[2, 2, 2, 2, 2, 2, 2])


pos_SF = nx.spring_layout(G_SF)
nx.draw(G_SF, pos_SF, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black", font_weight="bold")
plt.title("Directed Graph")
plt.show()

SFmodel.add_cpds(cpd_SF_01, cpd_SF_02, cpd_SF_03, cpd_SF_04, cpd_SF_05, cpd_SF_06, cpd_OF_for_SF, cpd_SF)
SFmodel.get_cpds()

inference_SF = VariableElimination(SFmodel)

query_result_SF = inference_SF.query(variables=['SF'])
values_SF=query_result_SF.values

prob_SF_0 = values_SF[0]
prob_SF_1 = values_SF[1]
# Printing query results
print('Probability of Supervisory factor Error is ', prob_SF_0)

'''Individual factors'''
'Alpha beta parameters for IF node'
alpha_IF_param=[4.5, 9.3, 6, 7, 4, 9, 5]
beta_IF_param=[2.1, 3.9, 2.5, 3, 2.8, 5, 2.2]

# Create a directed graph
G_IF = nx.DiGraph()

nodes_IF=['IF_01', 'IF_02', 'IF_03', 'IF_04', 'IF_05', 'IF_06', 'IF_07', 'OF_02', 'SF', 'IF']

G_IF.add_nodes_from(nodes_IF)

edges_IF=[('SF', 'IF'), ('OF_02', 'IF_02'), ('IF_01', 'IF'), ('IF_02', 'IF'), ('IF_03', 'IF'), ('IF_04', 'IF'), ('IF_05', 'IF'), ('IF_06', 'IF'), ('IF_07', 'IF')]
G_IF.add_edges_from(edges_IF)

beta_dist_IF_01 = beta(alpha_IF_param[0], beta_IF_param[0])
beta_dist_IF_02 = beta(alpha_IF_param[1], beta_IF_param[1])
beta_dist_IF_03 = beta(alpha_IF_param[2], beta_IF_param[2])
beta_dist_IF_04 = beta(alpha_IF_param[3], beta_IF_param[3])
beta_dist_IF_05 = beta(alpha_IF_param[4], beta_IF_param[4])
beta_dist_IF_06 = beta(alpha_IF_param[5], beta_IF_param[5])
beta_dist_IF_07 = beta(alpha_IF_param[6], beta_IF_param[6])

# Create subplots
fig_IF, axs_IF = plt.subplots(len(alpha_IF_param), 1, figsize=(8, 2 * len(alpha_IF_param)))

# Plot each beta distribution
for i in range(len(alpha_IF_param)):
    beta_dist = beta(alpha_IF_param[i], beta_IF_param[i])
    y_values = beta_dist.pdf(x_values)
    
    axs_IF[i].plot(x_values, y_values, label=f'Beta({alpha_IF_param[i]}, {beta_IF_param[i]})')
    axs_IF[i].set_title(f'Beta Distribution IF_{i+1}')
    axs_IF[i].legend()

# Adjust layout
plt.tight_layout()
plt.show()

'IF_01 parameters probability'
prob_IF_01_0 = beta_dist_IF_01.cdf(x[1]) - beta_dist_IF_01.cdf(x[0])
prob_IF_01_1 = beta_dist_IF_01.cdf(x[2]) - beta_dist_IF_01.cdf(x[1])

'IF_02 parameters probability'
prob_IF_02_0 = beta_dist_IF_02.cdf(x[1]) - beta_dist_IF_02.cdf(x[0])
prob_IF_02_1 = beta_dist_IF_02.cdf(x[2]) - beta_dist_IF_02.cdf(x[1])

'IF_03 parameters probability'
prob_IF_03_0 = beta_dist_IF_03.cdf(x[1]) - beta_dist_IF_03.cdf(x[0])
prob_IF_03_1 = beta_dist_IF_03.cdf(x[2]) - beta_dist_IF_03.cdf(x[1])

'IF_04 parameters probability'
prob_IF_04_0 = beta_dist_IF_04.cdf(x[1]) - beta_dist_IF_04.cdf(x[0])
prob_IF_04_1 = beta_dist_IF_04.cdf(x[2]) - beta_dist_IF_04.cdf(x[1])

'IF_05 parameters probability'
prob_IF_05_0 = beta_dist_IF_05.cdf(x[1]) - beta_dist_IF_05.cdf(x[0])
prob_IF_05_1 = beta_dist_IF_05.cdf(x[2]) - beta_dist_IF_05.cdf(x[1])

'IF_06 parameters probability'
prob_IF_06_0 = beta_dist_IF_06.cdf(x[1]) - beta_dist_IF_06.cdf(x[0])
prob_IF_06_1 = beta_dist_IF_06.cdf(x[2]) - beta_dist_IF_06.cdf(x[1])

'IF_07 parameters probability'
prob_IF_07_0 = beta_dist_IF_07.cdf(x[1]) - beta_dist_IF_07.cdf(x[0])
prob_IF_07_1 = beta_dist_IF_07.cdf(x[2]) - beta_dist_IF_07.cdf(x[1])

IFmodel = BayesianNetwork(edges_IF)

cpd_IF_01 = TabularCPD(variable='IF_01', variable_card=2, values=[[prob_IF_01_0], [prob_IF_01_1]])
cpd_IF_03 = TabularCPD(variable='IF_03', variable_card=2, values=[[prob_IF_03_0], [prob_IF_03_1]])
cpd_IF_04 = TabularCPD(variable='IF_04', variable_card=2, values=[[prob_IF_04_0], [prob_IF_04_1]])
cpd_IF_05 = TabularCPD(variable='IF_05', variable_card=2, values=[[prob_IF_05_0], [prob_IF_05_1]])
cpd_IF_06 = TabularCPD(variable='IF_06', variable_card=2, values=[[prob_IF_06_0], [prob_IF_06_1]])
cpd_IF_07 = TabularCPD(variable='IF_07', variable_card=2, values=[[prob_IF_07_0], [prob_IF_07_1]])

cpd_IF_02 = TabularCPD(variable='IF_02', variable_card=2, 
                       values=[[1, (1-min(prob_OF_02_1, prob_IF_02_1))], [0, min(prob_OF_02_1, prob_IF_02_1)]], 
                       evidence=['OF_02'], evidence_card=[2])
cpd_SF_for_IF = TabularCPD(variable='SF', variable_card=2, values=[[prob_SF_0], [prob_SF_1]])

IF_0_values=[1, prob_IF_01_0, prob_IF_02_0, 1-min(prob_IF_01_1, prob_IF_02_1), prob_IF_03_0, 1-min(prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_IF_04_0, 1-min(prob_IF_04_1, prob_IF_01_1), 1-min(prob_IF_04_1, prob_IF_02_1), 1-min(prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_IF_04_1, prob_IF_03_1), 1-min(prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_IF_05_0, 1-min(prob_IF_05_1, prob_IF_01_1), 1-min(prob_IF_05_1, prob_IF_02_1), 1-min(prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_IF_05_1, prob_IF_03_1), 1-min(prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_05_1, prob_IF_04_1), 1-min(prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 
             1-min(prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_IF_06_0, 
             1-min(prob_IF_06_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_02_1), 1-min(prob_IF_06_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_03_1), 1-min(prob_IF_06_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_06_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_IF_06_1, prob_IF_04_1), 1-min(prob_IF_06_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_IF_06_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 
             1-min(prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_05_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_02_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_03_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 
             1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_IF_07_0, 1-min(prob_IF_07_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_03_1), 1-min(prob_IF_07_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_IF_07_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_04_1), 1-min(prob_IF_07_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 
             1-min(prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_05_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_03_1), 
             1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 
             1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_IF_07_1, prob_IF_06_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_03_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1), 
             1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_01_1), 
             1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 
             1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_SF_0, 1-min(prob_SF_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_SF_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_04_1), 1-min(prob_SF_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 
             1-min(prob_SF_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_05_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_03_1), 
             1-min(prob_SF_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_04_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 
             1-min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_SF_1, prob_IF_06_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_SF_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_04_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1), 
             1-min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1), 
             1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1), 
             1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 
             1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_03_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_04_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_02_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_01_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_01_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 
             1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 1-min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 0]

#print(IF_0_values)
IF_1_values=[0, prob_IF_01_1, prob_IF_02_1, min(prob_IF_01_1, prob_IF_02_1), prob_IF_03_1, min(prob_IF_03_1, prob_IF_01_1), min(prob_IF_03_1, prob_IF_02_1), min(prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_IF_04_1, min(prob_IF_04_1, prob_IF_01_1), min(prob_IF_04_1, prob_IF_02_1), min(prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_IF_04_1, prob_IF_03_1), min(prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_IF_05_1, min(prob_IF_05_1, prob_IF_01_1), min(prob_IF_05_1, prob_IF_02_1), min(prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_IF_05_1, prob_IF_03_1), min(prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_05_1, prob_IF_04_1), min(prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), min(prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 
             min(prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), min(prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_IF_06_1, 
             min(prob_IF_06_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_02_1), min(prob_IF_06_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_03_1), min(prob_IF_06_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_06_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_IF_06_1, prob_IF_04_1), min(prob_IF_06_1, prob_IF_04_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_04_1, prob_IF_02_1), min(prob_IF_06_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_04_1, prob_IF_03_1), min(prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 
             min(prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_05_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_02_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_IF_06_1, prob_IF_05_1, prob_IF_03_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), 
             min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_IF_07_1, min(prob_IF_07_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_03_1), min(prob_IF_07_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_IF_07_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_04_1), min(prob_IF_07_1, prob_IF_04_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_04_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_04_1, prob_IF_03_1), min(prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 
             min(prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_05_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_03_1), 
             min(prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 
             min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_IF_07_1, prob_IF_06_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_03_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1), 
             min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_01_1), 
             min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 
             min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), prob_SF_1, min(prob_SF_1, prob_IF_01_1), min(prob_SF_1, prob_IF_02_1), min(prob_SF_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_03_1), min(prob_SF_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_SF_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_04_1), min(prob_SF_1, prob_IF_04_1, prob_IF_01_1), min(prob_SF_1, prob_IF_04_1, prob_IF_02_1), min(prob_SF_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_04_1, prob_IF_03_1), min(prob_SF_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 
             min(prob_SF_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_05_1), min(prob_SF_1, prob_IF_05_1, prob_IF_01_1), min(prob_SF_1, prob_IF_05_1, prob_IF_02_1), min(prob_SF_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_05_1, prob_IF_03_1), 
             min(prob_SF_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_05_1, prob_IF_04_1), min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), 
             min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_SF_1, prob_IF_06_1), min(prob_SF_1, prob_IF_06_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_02_1), min(prob_SF_1, prob_IF_06_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_03_1), min(prob_SF_1, prob_IF_06_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_SF_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_04_1), min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1), min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1), 
             min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1), 
             min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1), 
             min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), 
             min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1), min(prob_SF_1, prob_IF_07_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_03_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_04_1), min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_03_1), min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_02_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_03_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_03_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_01_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_01_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_03_1, prob_IF_02_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_02_1, prob_IF_01_1), 
             min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_01_1), min(prob_SF_1, prob_IF_07_1, prob_IF_06_1, prob_IF_05_1, prob_IF_04_1, prob_IF_03_1, prob_IF_02_1), 1]
#print(IF_1_values)
'''for i in range(len(IF_0_values)):
    IF_total = IF_0_values[i] + IF_1_values[i]
    print(i, IF_total)
'''

cpd_IF = TabularCPD(variable='IF', variable_card=2, values=[IF_0_values, IF_1_values], evidence=['IF_01', 'IF_02', 'IF_03', 'IF_04', 'IF_05', 'IF_06', 'IF_07', 'SF'], evidence_card=[2, 2, 2, 2, 2, 2, 2, 2])

pos_IF = nx.spring_layout(G_IF)
nx.draw(G_IF, pos_IF, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black", font_weight="bold")
plt.title("Directed Graph")
plt.show()

IFmodel.add_cpds(cpd_OF_02, cpd_IF_01, cpd_IF_02, cpd_IF_03, cpd_IF_04, cpd_IF_05, cpd_IF_06, cpd_IF_07, cpd_SF_for_IF, cpd_IF)
IFmodel.get_cpds()

inference_IF = VariableElimination(IFmodel)

query_result_IF = inference_IF.query(variables=['IF'])
values_IF=query_result_IF.values

prob_IF_0 = values_IF[0]
prob_IF_1 = values_IF[1]
# Printing query results
print('Probability of Individual factor Error is ', prob_IF_0)

'''Technical factors'''
'Alpha beta parameters for TF node'
alpha_TF_param=[12.5, 5, 8, 7.5, 8, 10]
beta_TF_param=[3.1, 3.1, 2.5, 3.9, 2.8, 2.5]

# Create a directed graph
G_TF = nx.DiGraph()

nodes_TF=['TF_01', 'TF_02', 'TF_03', 'TF_04', 'TF_05', 'SI2', 'IF', 'TF']

G_TF.add_nodes_from(nodes_TF)

edges_TF=[('TF_01', 'TF'), ('TF_02', 'TF'), ('TF_03', 'TF'), ('TF_04', 'TF'), ('TF_05', 'TF'),
          ('SI2', 'TF_01'), ('SI2', 'TF_02'), ('SI2', 'TF_03'), ('SI2', 'TF_05'), ('IF', 'TF_04')]
G_TF.add_edges_from(edges_TF)

beta_dist_TF_01 = beta(alpha_TF_param[0], beta_TF_param[0])
beta_dist_TF_02 = beta(alpha_TF_param[1], beta_TF_param[1])
beta_dist_TF_03 = beta(alpha_TF_param[2], beta_TF_param[2])
beta_dist_TF_04 = beta(alpha_TF_param[3], beta_TF_param[3])
beta_dist_TF_05 = beta(alpha_TF_param[4], beta_TF_param[4])
beta_dist_SI2 = beta(alpha_TF_param[5], beta_TF_param[5])

# Create subplots
fig_TF, axs_TF = plt.subplots(len(alpha_TF_param), 1, figsize=(8, 2 * len(alpha_TF_param)))

# Plot each beta distribution
for i in range(len(alpha_TF_param)):
    beta_dist = beta(alpha_TF_param[i], beta_TF_param[i])
    y_values = beta_dist.pdf(x_values)
    
    axs_TF[i].plot(x_values, y_values, label=f'Beta({alpha_TF_param[i]}, {beta_TF_param[i]})')
    axs_TF[i].set_title(f'Beta Distribution TF_{i+1}')
    axs_TF[i].legend()

# Adjust layout
plt.tight_layout()
plt.show()

'TF_01 parameters probability'
prob_TF_01_0 = beta_dist_TF_01.cdf(x[1]) - beta_dist_TF_01.cdf(x[0])
prob_TF_01_1 = beta_dist_TF_01.cdf(x[2]) - beta_dist_TF_01.cdf(x[1])

'TF_02 parameters probability'
prob_TF_02_0 = beta_dist_TF_02.cdf(x[1]) - beta_dist_TF_02.cdf(x[0])
prob_TF_02_1 = beta_dist_TF_02.cdf(x[2]) - beta_dist_TF_02.cdf(x[1])

'TF_03 parameters probability'
prob_TF_03_0 = beta_dist_TF_03.cdf(x[1]) - beta_dist_TF_03.cdf(x[0])
prob_TF_03_1 = beta_dist_TF_03.cdf(x[2]) - beta_dist_TF_03.cdf(x[1])

'TF_04 parameters probability'
prob_TF_04_0 = beta_dist_TF_04.cdf(x[1]) - beta_dist_TF_04.cdf(x[0])
prob_TF_04_1 = beta_dist_TF_04.cdf(x[2]) - beta_dist_TF_04.cdf(x[1])

'TF_05 parameters probability'
prob_TF_05_0 = beta_dist_TF_05.cdf(x[1]) - beta_dist_TF_05.cdf(x[0])
prob_TF_05_1 = beta_dist_TF_05.cdf(x[2]) - beta_dist_TF_05.cdf(x[1])

'Security incidence impact distribution'
prob_SI2_0 = beta_dist_SI2.cdf(x[1]) - beta_dist_SI2.cdf(x[0])
prob_SI2_1 = beta_dist_SI2.cdf(x[2]) - beta_dist_SI2.cdf(x[1])

TFmodel = BayesianNetwork(edges_TF)

cpd_SI2 = TabularCPD(variable='SI2', variable_card=2, values=[[prob_SI2_0], [prob_SI2_1]])
cpd_IF_for_TF_04 = TabularCPD(variable='IF', variable_card=2, values=[[prob_IF_0], [prob_IF_1]])

cpd_TF_01 = TabularCPD(variable='TF_01', variable_card=2, 
                       values=[[1, (1-min(prob_SI2_1, prob_TF_01_1))], [0, min(prob_SI2_1, prob_TF_01_1)]], 
                       evidence=['SI2'], evidence_card=[2])
cpd_TF_02 = TabularCPD(variable='TF_02', variable_card=2, 
                       values=[[1, (1-min(prob_SI2_1, prob_TF_02_1))], [0, min(prob_SI2_1, prob_TF_02_1)]], 
                       evidence=['SI2'], evidence_card=[2])
cpd_TF_03 = TabularCPD(variable='TF_03', variable_card=2, 
                       values=[[1, (1-min(prob_SI2_1, prob_TF_03_1))], [0, min(prob_SI2_1, prob_TF_03_1)]], 
                       evidence=['SI2'], evidence_card=[2])
cpd_TF_05 = TabularCPD(variable='TF_05', variable_card=2, 
                       values=[[1, (1-min(prob_SI2_1, prob_TF_05_1))], [0, min(prob_SI2_1, prob_TF_05_1)]], 
                       evidence=['SI2'], evidence_card=[2])

cpd_TF_04 = TabularCPD(variable='TF_04', variable_card=2, 
                       values=[[1, (1-min(prob_IF_1, prob_TF_04_1))], [0, min(prob_IF_1, prob_TF_04_1)]], 
                       evidence=['IF'], evidence_card=[2])



TF_0_values=[1, prob_TF_01_0, prob_TF_02_0, 1-min(prob_TF_01_1, prob_TF_02_1), prob_TF_03_0, 1-min(prob_TF_03_1, prob_TF_01_1), 1-min(prob_TF_03_1, prob_TF_02_1), 1-min(prob_TF_03_1, prob_TF_02_1, prob_TF_01_1), prob_TF_04_0, 1-min(prob_TF_04_1, prob_TF_01_1), 1-min(prob_TF_04_1, prob_TF_02_1), 1-min(prob_TF_04_1, prob_TF_02_1, prob_TF_01_1), 1-min(prob_TF_04_1, prob_TF_03_1), 
             1-min(prob_TF_04_1, prob_TF_03_1, prob_TF_01_1), 1-min(prob_TF_04_1, prob_TF_03_1, prob_TF_02_1), 1-min(prob_TF_04_1, prob_TF_03_1, prob_TF_02_1, prob_TF_01_1), prob_TF_05_0, 1-min(prob_TF_05_1, prob_TF_01_1), 1-min(prob_TF_05_1, prob_TF_02_1), 1-min(prob_TF_05_1, prob_TF_02_1, prob_TF_01_1), 1-min(prob_TF_05_1, prob_TF_03_1), 1-min(prob_TF_05_1, prob_TF_03_1, prob_TF_01_1), 
             1-min(prob_TF_05_1, prob_TF_03_1, prob_TF_02_1), 1-min(prob_TF_05_1, prob_TF_03_1, prob_TF_02_1, prob_TF_01_1), 1-min(prob_TF_05_1, prob_TF_04_1), 1-min(prob_TF_05_1, prob_TF_04_1, prob_TF_01_1), 1-min(prob_TF_05_1, prob_TF_04_1, prob_TF_02_1), 1-min(prob_TF_05_1, prob_TF_04_1, prob_TF_02_1, prob_TF_01_1), 1-min(prob_TF_05_1, prob_TF_04_1, prob_TF_03_1), 
             1-min(prob_TF_05_1, prob_TF_04_1, prob_TF_03_1, prob_TF_01_1), 1-min(prob_TF_05_1, prob_TF_04_1, prob_TF_03_1, prob_TF_02_1), 0]

TF_1_values=[0, prob_TF_01_1, prob_TF_02_1, min(prob_TF_01_1, prob_TF_02_1), prob_TF_03_1, min(prob_TF_03_1, prob_TF_01_1), min(prob_TF_03_1, prob_TF_02_1), min(prob_TF_03_1, prob_TF_02_1, prob_TF_01_1), prob_TF_04_1, min(prob_TF_04_1, prob_TF_01_1), min(prob_TF_04_1, prob_TF_02_1), min(prob_TF_04_1, prob_TF_02_1, prob_TF_01_1), min(prob_TF_04_1, prob_TF_03_1), 
             min(prob_TF_04_1, prob_TF_03_1, prob_TF_01_1), min(prob_TF_04_1, prob_TF_03_1, prob_TF_02_1), min(prob_TF_04_1, prob_TF_03_1, prob_TF_02_1, prob_TF_01_1), prob_TF_05_1, min(prob_TF_05_1, prob_TF_01_1), min(prob_TF_05_1, prob_TF_02_1), min(prob_TF_05_1, prob_TF_02_1, prob_TF_01_1), min(prob_TF_05_1, prob_TF_03_1), min(prob_TF_05_1, prob_TF_03_1, prob_TF_01_1), 
             min(prob_TF_05_1, prob_TF_03_1, prob_TF_02_1), min(prob_TF_05_1, prob_TF_03_1, prob_TF_02_1, prob_TF_01_1), min(prob_TF_05_1, prob_TF_04_1), min(prob_TF_05_1, prob_TF_04_1, prob_TF_01_1), min(prob_TF_05_1, prob_TF_04_1, prob_TF_02_1), min(prob_TF_05_1, prob_TF_04_1, prob_TF_02_1, prob_TF_01_1), min(prob_TF_05_1, prob_TF_04_1, prob_TF_03_1), 
             min(prob_TF_05_1, prob_TF_04_1, prob_TF_03_1, prob_TF_01_1), min(prob_TF_05_1, prob_TF_04_1, prob_TF_03_1, prob_TF_02_1), 1]

cpd_TF = TabularCPD(variable='TF', variable_card=2, values=[TF_0_values, TF_1_values], evidence=['TF_01', 'TF_02', 'TF_03', 'TF_04', 'TF_05'], evidence_card=[2, 2, 2, 2, 2])

pos_TF = nx.spring_layout(G_TF)
nx.draw(G_TF, pos_TF, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black", font_weight="bold")
plt.title("Directed Graph")
plt.show()

TFmodel.add_cpds(cpd_TF_01, cpd_TF_02, cpd_TF_03, cpd_TF_04, cpd_TF_05, cpd_SI2, cpd_IF_for_TF_04, cpd_TF)
TFmodel.get_cpds()

inference_TF = VariableElimination(TFmodel)

query_result_TF = inference_TF.query(variables=['TF'])
values_TF=query_result_TF.values

prob_TF_0 = values_TF[0]
prob_TF_1 = values_TF[1]
# Printing query results
print('Probability of Technical factor Error is ', prob_TF_0)

'''Human error BBN model'''
G = nx.DiGraph()
# Define nodes
nodes = ['HE', 'OF', 'SF', 'IF', 'TF']

# Add nodes to the graph
G.add_nodes_from(nodes)

# Add edges from OF, SF, IF, and TF to HE
edges = [('OF', 'HE'), ('SF', 'HE'), ('IF', 'HE'), ('TF', 'HE')]
G.add_edges_from(edges)
HEmodel = BayesianNetwork(edges)

cpd_OF_for_HE = TabularCPD(variable='OF', variable_card=2, values=[[prob_OF_0], [prob_OF_1]])
cpd_SF_for_HE = TabularCPD(variable='SF', variable_card=2, values=[[prob_SF_0], [prob_SF_1]])
cpd_IF_for_HE = TabularCPD(variable='IF', variable_card=2, values=[[prob_IF_0], [prob_IF_1]])
cpd_TF_for_HE = TabularCPD(variable='TF', variable_card=2, values=[[prob_TF_0], [prob_TF_1]])

HE_0_values=[1, prob_TF_0, prob_IF_0, (1-(min(prob_IF_1, prob_TF_1))), 
           prob_SF_0, (1-(min(prob_SF_1, prob_TF_1))), (1-(min(prob_SF_1, prob_IF_1))), (1-(min(prob_SF_1, prob_TF_1, prob_IF_1))), 
           prob_OF_0, (1-(min(prob_OF_1, prob_TF_1))), (1-(min(prob_OF_1, prob_SF_1, prob_TF_1))), (1-(min(prob_OF_1, prob_TF_1, prob_IF_1))), 
           (1-(min(prob_OF_1, prob_SF_1))), (1-(min(prob_OF_1, prob_SF_1, prob_TF_1))), (1-min(prob_OF_1, prob_SF_1, prob_IF_1)), 0]
HE_1_values=[0, prob_TF_1, prob_IF_1, min(prob_IF_1, prob_TF_1), 
           prob_SF_1, min(prob_SF_1, prob_TF_1), min(prob_SF_1, prob_IF_1), min(prob_SF_1, prob_TF_1, prob_IF_1),
           prob_OF_1, min(prob_OF_1, prob_TF_1), min(prob_OF_1, prob_SF_1, prob_TF_1), min(prob_OF_1, prob_TF_1, prob_IF_1),
          min(prob_OF_1, prob_SF_1), min(prob_OF_1, prob_SF_1, prob_TF_1), min(prob_OF_1, prob_SF_1, prob_IF_1), 1]

cpd_HE = TabularCPD(variable='HE', variable_card=2, 
                    values=[HE_0_values, HE_1_values], 
                    evidence=['OF', 'SF', 'IF', 'TF'], evidence_card=[2, 2, 2, 2])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black", font_weight="bold")
plt.title("Directed Graph")
plt.show()

HEmodel.add_cpds(cpd_OF_for_HE, cpd_SF_for_HE, cpd_IF_for_HE, cpd_TF_for_HE, cpd_HE)
HEmodel.get_cpds()

inference = VariableElimination(HEmodel)

query_result_HE = inference.query(variables=['HE'])
values_HE=query_result_HE.values
# Printing query results
print('Probability of Human Error is ', values_HE[0])

'''
plt.bar(range(len(values_HE)), values_HE, color='skyblue')
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Probability of Human Error')

plt.xticks(range(len(values_TF)), ['Error=True', 'Error=False'])
plt.show()

plt.bar(range(len(values_TF)), values_TF, color='skyblue')
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Probability of Technical Error')

plt.xticks(range(len(values_TF)), ['Error=True', 'Error=False'])
plt.show()

plt.bar(range(len(values_IF)), values_IF, color='skyblue')
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Probability of Individual Error')

plt.xticks(range(len(values_IF)), ['Error=True', 'Error=False'])
plt.show()

plt.bar(range(len(values_SF)), values_SF, color='skyblue')
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Probability of Supervisory Error')

plt.xticks(range(len(values_SF)), ['Error=True', 'Error=False'])
plt.show()

plt.bar(range(len(values_OF)), values_OF, color='skyblue')
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Probability of Organisational Error')

plt.xticks(range(len(values_OF)), ['Error=True', 'Error=False'])
plt.show()
'''

fig, axs = plt.subplots(5, 1, figsize=(8, 12))

# Plot Probability of Human Error
axs[0].bar(range(len(values_OF)), values_HE, color='skyblue')
axs[0].set_xlabel('Error')
axs[0].set_ylabel('Probability')
axs[0].set_title('Probability of Organisational Error')
axs[0].set_xticks(range(len(values_OF)))
axs[0].set_xticklabels(['Error=True', 'Error=False'])

# Plot Probability of Technical Error
axs[1].bar(range(len(values_TF)), values_TF, color='skyblue')
axs[1].set_xlabel('Error')
axs[1].set_ylabel('Probability')
axs[1].set_title('Probability of Technical Error')
axs[1].set_xticks(range(len(values_TF)))
axs[1].set_xticklabels(['Error=True', 'Error=False'])

# Plot Probability of Individual Error
axs[2].bar(range(len(values_IF)), values_IF, color='skyblue')
axs[2].set_xlabel('Error')
axs[2].set_ylabel('Probability')
axs[2].set_title('Probability of Individual Error')
axs[2].set_xticks(range(len(values_IF)))
axs[2].set_xticklabels(['Error=True', 'Error=False'])

# Plot Probability of Supervisory Error
axs[3].bar(range(len(values_SF)), values_SF, color='skyblue')
axs[3].set_xlabel('Error')
axs[3].set_ylabel('Probability')
axs[3].set_title('Probability of Supervisory Error')
axs[3].set_xticks(range(len(values_SF)))
axs[3].set_xticklabels(['Error=True', 'Error=False'])

# Plot Probability of Organisational Error
axs[4].bar(range(len(values_HE)), values_OF, color='skyblue')
axs[4].set_xlabel('Error')
axs[4].set_ylabel('Probability')
axs[4].set_title('Probability of Human Error')
axs[4].set_xticks(range(len(values_HE)))
axs[4].set_xticklabels(['Error=True', 'Error=False'])

# Adjust layout
plt.tight_layout()
plt.show()