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
		
class DecantationSearchProblem(SearchProblem):
	def __init__(self, capacities, starting_amts, goal_amt):
		self.capacities = capacities
		self.starting_amts = starting_amts
		self.goal_amt = goal_amt
	
	def start_node(self):
		return self.starting_amts
	
	def is_goal(self, node):
		return self.goal_amt in node
	
	def neighbors(self, node):
		neighbors = []
		for i in range(len(node)):
			for j in range(len(node)):
				if i == j:
					continue
				
				transfer_amt = min(node[i], self.capacities[j] - node[j])
				
				if transfer_amt == 0:
					continue

				new_node = list(node)
				new_node[i] = node[i] - transfer_amt
				new_node[j] = node[j] + transfer_amt
				neighbors.append(Edge(node, tuple(new_node)))
		return neighbors
		
			
	
	
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
	
class BFSSearcher(Searcher):
	def __init__(self, problem):
		self.problem = problem
		
	def search(self):
		frontier = [Path(self.problem.start_node())]
		visited = set(frontier)
		while len(frontier) > 0:
			path = frontier.pop()
			last_node,_ = path.peek()
			if self.problem.is_goal(last_node):
				yield path
			neighbors = self.problem.neighbors(last_node)
			for edge in neighbors:
				neighbor = edge.v
				if neighbor in visited:
					continue
				visited.add(neighbor)
				new_path = copy.deepcopy(path)
				new_path.add(neighbor, 1)
				frontier.insert(0, new_path)		
		
		
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

problem = DecantationSearchProblem((8, 5, 3), (8, 0, 0), 4)
searcher = BFSSearcher(problem)
for path in searcher.search():
	print(path)

