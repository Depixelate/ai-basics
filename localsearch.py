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
		
class NQueensSearchProblem(SearchProblem):
	def __init__(self, n):
		self.n = n
	
	def start_node(self):
		return [[x, random.randrange(0, self.n)] for x in range(self.n)]
	
	def is_goal(self, node):
		return self.heuristic(node) == 0
	
	def neighbors(self, node):
		neighbors = []
		for i in range(self.n):
			for j in range(self.n):
				if j == node[i][1]:
					continue
				new_node = copy.deepcopy(node)
				new_node[i][1] = j
				neighbors.append(Edge(node, new_node))
		return neighbors
	
	def heuristic(self, node):
		num_queens_attacking = 0
		for i in range(self.n):
			for j in range(i+1, self.n):
				q1x, q1y, q2x, q2y = node[i][0], node[i][1], node[j][0], node[j][1]
				if q1x == q2x or q1y == q2y or abs(q2x - q1x) == abs(q2y - q1y):
					num_queens_attacking += 1
					
		return num_queens_attacking
	
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

class RandomRestartHillClimbingSearcher(Searcher):
	
	def __init__(self, problem):
		self.problem = problem
	
	def singleHillClimb(self):
		cur_node = self.problem.start_node()
		history = [(cur_node, self.problem.heuristic(cur_node))]
		while True:
			next_node = min(self.problem.neighbors(cur_node), key=lambda edge: self.problem.heuristic(edge.v)).v
			h_val = self.problem.heuristic(next_node)
			if self.problem.heuristic(next_node) >= self.problem.heuristic(cur_node):
				break
			cur_node = next_node
			history.append((next_node,  h_val))
		return cur_node, history
			
	
	def search(self, max_climbs=1000):
		climbs = 0
		while climbs < max_climbs:
			res_node, history = self.singleHillClimb()
			h_val = self.problem.heuristic(res_node)
			print(h_val)
			if h_val == 0:
				break
			climbs += 1
		return res_node, history


searcher = RandomRestartHillClimbingSearcher(NQueensSearchProblem(8))

res_node, history = searcher.search()

print(f"Final State: {res_node}")
print()
print("History: ")
for node,heuristic in history:
	print(f"{node} - {heuristic}")



