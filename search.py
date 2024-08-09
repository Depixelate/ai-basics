from abc import ABC abstractmethod

class Edge:
	def __init__(self, u, v, cost=1):
		self.u = u
		self.v = v
		self.cost = cost
	
	def __repr__(self):
		return f"{u}--{cost}--{v}


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
	def __init__(self, edges, start, goals):
		self.edges = edges
		self.start = start
		self.goals = goals
	
	def start_node(self):
		return self.start
	
	def is_goal(self, node):
		return node in self.goals
	
	def neighbors(self, node):
		neighbors = [edge.u for edge in self.edges if edge.v == node] + [edge.v for edge in self.edges if edge.u == node]
		return neighbors
	
	
class Path:
	def __init__(self, initial, rem_path = None):
		self.initial = initial
		self.rem_path = rem_path
		if rem_path is None:
			self.cost = 0
		else:
			self.cost = initial.cost + rem_path.cost
		
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
			return self.initial
		return f"{self.initial.u}-{self.rem_path}"
	
problem1 = Search_problem_from_explicit_graph(edges=
[Edge(’A’,’B’,3), Edge(’A’,’C’,1), Edge(’B’,’D’,1), Edge(’B’,’G’,3),
Edge(’C’,’B’,1), Edge(’C’,’D’,3), Edge(’D’,’G’,1)],
start = ’A’, goals = {’G’})

class Searcher(ABC):
	@abstractmethod
	def search(self):
		pass

class DFSSearcher(Searcher):
	def __init__(self, problem):
		self.problem = problem
		self.stack = [[problem.start_node()]]
		self.visited = set()
	
	def search_helper(self):
		if self.problem.is_goal(self.stack[-1]):
			yield self.stack
		else:
			neighbors = self.problem.neighbors(self.stack[-1])
			for neighbor in neighbors:
				if neighbor in 
	
	def search(self):
		while True:	
			if len(self.stack) == 0:
				return None
			
			yield search_helper()
				
			
			
	
		
		
		
	
		
		
		
		
