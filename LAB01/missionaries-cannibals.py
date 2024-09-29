from collections import deque

def is_valid(state):
    missionaries, cannibals, boat = state
    if missionaries < 0 or cannibals < 0 or missionaries > 3 or cannibals > 3:
        return False
    if missionaries > 0 and missionaries < cannibals:
        return False
    if 3 - missionaries > 0 and 3 - missionaries < 3 - cannibals:
        return False
    return True

def get_successors(state):
    successors = []
    missionaries, cannibals, boat = state
    if boat == 1:
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        for move in moves:
            new_state = (missionaries - move[0], cannibals - move[1], 0)
            if is_valid(new_state):
                successors.append(new_state)
    else:
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        for move in moves:
            new_state = (missionaries + move[0], cannibals + move[1], 1)
            if is_valid(new_state):
                successors.append(new_state)
    return successors

# implementing bfs using queue
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
    return None, len(visited), max_queue_size

# implementing dfs using stack
def dfs(start_state, goal_state):
    stack = [(start_state, [])]
    visited = set()
    max_stack_size = 1  
    while stack:
        (state, path) = stack.pop() 
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            return path, len(visited), max_stack_size  
        for successor in get_successors(state):
            stack.append((successor, path))
        max_stack_size = max(max_stack_size, len(stack)) 
    return None, len(visited), max_stack_size
    
# start state   
start_state = (3, 3, 1)
# goal state
goal_state = (0, 0, 0)

# Using BFS
solution, length, max_queue_size = bfs(start_state, goal_state)
print("Using BFS: ")
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
    print("Number of steps to reach the solution: ", len(solution)-1)
    print("Different states visited:", length)
    print("Maximum queue size:", max_queue_size)
else:
    print("No solution found.")
    

print("------------------------------------")
# Using dfs
print("Using DFS: ")
solution, length, max_stack_size = dfs(start_state, goal_state)

if solution:
    print("Solution found:")
    for step in solution:
        print(step)
    print("Number of steps to reach to solution: ", len(solution)-1)
    print("Different states visited:", length)
    print("Maximum stack size:", max_stack_size)
else:
    print("No solution found.")


    
