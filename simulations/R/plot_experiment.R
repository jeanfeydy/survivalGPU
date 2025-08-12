library(ggplot2)
library(dplyr)


df = read.csv("result_experiment_500.csv")



scenario_list = unique(df$scenario_name)
print(scenario_list)

for(sc)

