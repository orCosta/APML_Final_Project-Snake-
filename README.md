# APML_Final_Project-Snake

#info
Create different kind of learning agents for classic Snake game.
Agent 1: Linear approximation, taking the 4 optional next actions as a vector, and learning and calculating the best weights vector.

Agent 2: Using NN model, taking the 4 optional next actions as a vector, and pass them through small FC NN that return prediction for the best next move.

Agent 3:  Using deep q learning technique, saving Q value for each state that represent the best action to apply next. Calculate and update weights vector using SGD, and save some move history for every update. 
