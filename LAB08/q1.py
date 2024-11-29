wall = [(1,1)]
terminal_states = [(1,3),(2,3)]

# possible actions (equally probable)
action_probability = {'L':0.25,'R':0.25,'U':0.25,'D':0.25}

# stochastic environment actions
environment_left = {'L':'D','R':'U','U':'L','D':'R'}
environment_right = {'L':'U','R':'D','U':'R','D':'L'}

# cell is available or not
def is_valid(i,j):
    return (i,j) not in wall and i >= 0 and i < 3 and j >= 0 and j < 4
    
#take action
def transition(action,i,j):
    if action == 'L':
        return (i,j-1)
    elif action == 'R':
        return (i,j+1)
    elif action == 'U':
        return (i+1,j)
    elif action == 'D':
        return (i-1,j)   
    else:
        return (-1,-1)
        
        
# value function
def value_function(i,j,reward,reward_matrix,discount_factor,V_pie):
    value = 0
    for action in ['L','R','U','D']:
        # desired action with 0.8 probability
        state_x,state_y = transition(action,i,j)
        if is_valid(state_x,state_y):
            desired_action_value = (reward_matrix[state_x][state_y] + discount_factor*V_pie[state_x][state_y])
        else:
            desired_action_value = (reward_matrix[i][j] + discount_factor*V_pie[i][j])
        
        # environment action with 0.1 probability
        state_x,state_y = transition(environment_left[action],i,j)
        if is_valid(state_x,state_y):
            env_action_left_value = (reward_matrix[state_x][state_y] + discount_factor*V_pie[state_x][state_y])
        else:
            env_action_left_value = (reward_matrix[i][j] + discount_factor*V_pie[i][j])
        
        # environment action with 0.1 probability 
        state_x,state_y = transition(environment_right[action],i,j)
        if is_valid(state_x,state_y):
            env_action_right_value = (reward_matrix[state_x][state_y] + discount_factor*V_pie[state_x][state_y])
        else:
            env_action_right_value = (reward_matrix[i][j] + discount_factor*V_pie[i][j])
        
        value_to_action = desired_action_value*0.8+env_action_left_value*0.1+env_action_right_value*0.1        

        value += value_to_action*0.25

    return value

# iterative policy evaluation
def iterative_policy_evaluation(iter,epsilon,reward,reward_matrix,V_pie,discount_factor):
    while True:
        delta = 0
        for i in range(3):
            for j in range(4):
                state = (i,j)
                if state in terminal_states or state in wall:  # continue if encounter terminal state or wall
                    continue
                v = V_pie[i][j]
                V_pie[i][j] = value_function(i,j,reward,reward_matrix,discount_factor,V_pie)
                delta = max(delta,abs(v-V_pie[i][j]))
        iter += 1
        if delta < epsilon:
            print(f"Number of iterations = {iter}")
            break 
    print_values(V_pie)

# reward matrix i
def update_reward_matrix(reward):
    reward_matrix = [[reward for _ in range(4)] for _ in range(3)]
    reward_matrix[2][3] = 1
    reward_matrix[1][3] = -1
    return reward_matrix

# V_pie initialization
def initialize_V_pie():
    V_pie = [[0 for _ in range(4)]for _ in range(3)]
    return V_pie
    
#printing matrix after convergence 
def print_values(V):
  for i in range(2,-1,-1):
    print(" ")
    for j in range(4):
      v = V[i][j]
      print(" %.2f|" % v, end="")
    print("")

rewards = [-0.04,-2,0.1,0.02,1]
epsilon = 1e-8
discount_factor = 1
print("Value Functions corresponding to optimal policy\n")
for reward in rewards:
    print(f"For r(S) : {reward}")
    reward_matrix = update_reward_matrix(reward )
    V_pie = initialize_V_pie()
    iterative_policy_evaluation(0, epsilon, reward, reward_matrix, V_pie, discount_factor)
    print("\n")