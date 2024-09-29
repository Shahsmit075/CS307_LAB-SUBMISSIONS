from collections import deque

# start state
start_state = ('E', 'E', 'E', 'O', 'W', 'W', 'W')
# goal state
goal_state = ('W', 'W', 'W', 'O', 'E', 'E', 'E')

def is_valid(new_index, state):
    return 0 <= new_index < len(state)

def get_successors(state):
    successors = []
    empty_index = state.index('O')

    move_options = [-1, 1, -2, 2]
    
    for move in move_options:
        new_index = empty_index + move
        if is_valid(new_index, state):
            new_state = list(state)
            new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
            successors.append(tuple(new_state))
    
    return successors

# Implementing bfs
def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    max_queue_size = 1
    
    while queue:
        (state, path) = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        
        if state == goal_state:
            return path, len(visited), max_queue_size
        
        for successor in get_successors(state):
            queue.append((successor, path))
        max_queue_size = max(max_queue_size, len(queue))  
    
    return None, 0, 0

# Implementing dfs  
def dfs(start_state, goal_state):
    queue = [(start_state, [])]
    visited = set()
    max_queue_size = 1  
    
    while queue:
        (state, path) = queue.pop()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        
        if state == goal_state:
            return path, len(visited), max_queue_size
        
        for successor in get_successors(state):
            queue.append((successor, path))
        max_queue_size = max(max_queue_size, len(queue))  
    
    return None, 0, 0


# Finding solution using bfs
solution, visited_length, max_queue_size = bfs(start_state, goal_state)
print("Using BFS: ")
print("Total Number of operations:", len(solution) - 1 if solution else 0)
print("Maximum queue size:", max_queue_size)
print("Number of visited nodes:", visited_length)
print()

if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
    

print("-----------------------------------------------")

# Finding solution uing dfs
solution, visited_length, max_queue_size = dfs(start_state, goal_state)
print("Using DFS: ")
print("Total Number of operations:", len(solution) - 1 if solution else 0)
print("Maximum queue size:", max_queue_size)
print("Number of visited nodes:", visited_length)
print()

if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")