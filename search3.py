from abc import ABC, abstractmethod
import heapq
import copy
import math
import random

class Edge:
	def __init__(self, u, v, cost=1):
		self.u = u
		self.v = v
		self.cost = cost
	
	def __repr__(self):
		return f"{self.u}--{self.cost}--{self.v}"

	def reverse(self):
		return Edge(self.v, self.u, self.cost)

def add_reverse(edges):
	revs = [Edge(edge.v, edge.u, edge.cost) for edge in edges]
	return edges + revs
	

class SearchProblem(ABC):
	
	@abstractmethod
	def start_node(self):
		pass
	
	@abstractmethod
	def is_goal(self, node):
		pass
	
	@abstractmethod
	def neighbors(self, node):
		pass
	
	def heuristic(self, node):
		return 0

	
class ExplicitGraphSearchProblem(SearchProblem):
	def __init__(self, edges, start, goals, heuristicDict=None):
		self.edges = edges
		self.start = start
		self.goals = goals
		self.heuristicDict = heuristicDict
	
	def start_node(self):
		return self.start
	
	def is_goal(self, node):
		return node in self.goals
	
	def neighbors(self, node):
		neighbors = [edge for edge in self.edges if edge.u == node]
		return neighbors
	
	def heuristic(self, node):
		if self.heuristicDict is None:
			return 0
		return self.heuristicDict[node]
	
	
class Path:
	def __init__(self, initial, rem_path = None):
		self.initial = initial
		self.rem_path = rem_path
		if rem_path is None:
			self.cost = 0
		else:
			self.cost = initial.cost + rem_path.cost
   
	def add(self, elem, cost):
		if self.rem_path is not None:
			self.rem_path.add(elem, cost)
			self.cost += cost
		else:
			self.initial = Edge(self.initial, elem, cost)
			self.cost = cost
			self.rem_path = Path(elem)
   
	def peek(self):
		if self.rem_path is not None:
			node, cost = self.rem_path.peek()
			if node == self.initial.u:
				cost = self.initial.cost
			return node, cost
		return self.initial, 0
  
	def remove(self):
		if self.rem_path is not None:
			node, cost = self.rem_path.remove()
			if self.initial.v == node:
				cost = self.initial.cost
				self.initial = self.initial.u
				self.rem_path = None
			self.cost -= cost
			return node, cost
		return self.initial, 0

		
	def next(self):
		return self.rem_path
	
	def start(self):
		return self.initial.u if self.rem_path is not None else self.initial
	
	def nodes(self):
		cur = self
		while cur is not None:
			yield cur.start
			cur = cur.next()
	
	def __repr__(self):
		if self.rem_path is None:
			return str(self.initial)
		return f"{self.initial.u}--{self.initial.cost}--{self.rem_path}"
	

class Searcher(ABC):
	@abstractmethod
	def search(self):
		pass

class FrontierPQ:
	def __init__(self):
		self.frontier_index = 0
		self.pqueue = []
	
	def push(self, priority, path):
		heapq.heappush(self.pqueue, (priority, self.frontier_index, path))
		self.frontier_index += 1
	
	def pop(self):
		return heapq.heappop(self.pqueue)[2]
	
	def is_empty(self):
		return len(self.pqueue) == 0
	
def misplaced_tiles_heuristic(state):
	num_misplaced = 0
	for i in range(3):
		for j in range(3):
			if state[i][j] != 3 * i + j + 1:
				num_misplaced += 1
	
	return num_misplaced

def manhattan_dist_heuristic(state):
	tot_dist = 0
	for i in range(3):
		for j in range(3):
			correct_i = (state[i][j] - 1) // 3
			correct_j = (state[i][j] - 1) % 3
			tot_dist += abs(i - correct_i) + abs(j - correct_j)
	
	return tot_dist
	
	
	
class Puzzle8Problem(SearchProblem):
	def __init__(self, array = None, h_func = manhattan_dist_heuristic): # 9 is the gap
		if array is None:
			array = list(range(1, 10))
			random.shuffle(array)
			self.start_state = tuple(tuple(array[3*i + j] for j in range(3)) for i in range(3))
			
		else:
			self.start_state = array
		
		self.h_func = h_func
	
	def start_node(self):
		return self.start_state
	
	def is_goal(self, state):
		for i in range(3):
			for j in range(3):
				if state[i][j] != 3 * i + j + 1:
					return False
		return True
	
	def neighbors(self, state):
		blank_i = 0
		blank_j = 0
		for i in range(3):
			for j in range(3):
				if state[i][j] == 9:
					blank_i = i
					blank_j = j
		options = []
		
		if blank_i - 1 >= 0:
			options.append((blank_i - 1, blank_j))
	
		if blank_i + 1 < 3:
			options.append((blank_i + 1, blank_j))
			
		if blank_j - 1 >= 0:
			options.append((blank_i, blank_j - 1))
	
		if blank_j + 1 < 3:
			options.append((blank_i, blank_j + 1))
			
		neighbors = []
		for i, j in options:
			new_state = [[state[i][j] for j in range(3)] for i in range(3)]
			new_state[i][j], new_state[blank_i][blank_j] = new_state[blank_i][blank_j], new_state[i][j]
			new_state = tuple(tuple(new_state[i][j] for j in range(3)) for i in range(3))
			neighbors.append(Edge(state, new_state))
		
		return neighbors
	
	def heuristic(self, state):
		return self.h_func(state)
				
			
class AStarSearcher(Searcher):
	def __init__(self, problem):
		self.problem = problem
		self.visited = set()
		self.frontier = FrontierPQ()
		self.frontier.push(self.problem.heuristic(self.problem.start_node()), Path(self.problem.start_node()))
	
	def search(self):
		while not self.frontier.is_empty():
			path = self.frontier.pop()
			last = path.peek()[0]
			if last in self.visited:
				continue
			self.visited.add(last)
			if self.problem.is_goal(last):
				yield path
			neighbors = self.problem.neighbors(last)
			for neighbor in neighbors:
				neighbor, cost = neighbor.v, neighbor.cost
				if neighbor in self.visited:
					continue
				new_path = copy.deepcopy(path)
				new_path.add(neighbor, cost)
				self.frontier.push(new_path.cost + self.problem.heuristic(neighbor), new_path)

class GreedyBestFirstSearcher(Searcher):
	def __init__(self, problem):
		self.problem = problem
		self.visited = set()
		self.frontier = FrontierPQ()
		self.frontier.push(self.problem.heuristic(self.problem.start_node()), Path(self.problem.start_node()))
		
	def search(self):
		while not self.frontier.is_empty():
			path = self.frontier.pop()
			last = path.peek()[0]
			if last in self.visited:
				continue
			self.visited.add(last)
			if self.problem.is_goal(last):
				yield path
			neighbors = self.problem.neighbors(last)
			for neighbor in neighbors:
				neighbor, cost = neighbor.v, neighbor.cost
				if neighbor in self.visited:
					continue
				new_path = copy.deepcopy(path)
				new_path.add(neighbor, cost)
				self.frontier.push(self.problem.heuristic(neighbor), new_path)
				

tiles_problem = (
(9, 1, 3),
(4, 2, 5),
(7, 8, 6)
)

astar_searcher = GreedyBestFirstSearcher(Puzzle8Problem(tiles_problem, misplaced_tiles_heuristic))

print("A Star: ")
for path in astar_searcher.search():
	print(path)	

