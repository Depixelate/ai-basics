from abc import ABC, abstractmethod
import heapq
import copy

class Edge:
	def __init__(self, u, v, cost=1):
		self.u = u
		self.v = v
		self.cost = cost
	
	def __repr__(self):
		return f"{self.u}--{self.cost}--{self.v}"

	def reverse(self):
		return Edge(self.v, self.u, self.cost)


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

	def __repr__(self):
		adj_list = {}
		for edge in self.edges:
			adj_list[edge.u] = adj_list.get(edge.u, []) + [edge.v]
		return repr(adj_list)

     
	
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
			return self.initial
		return f"{self.initial.u}--{self.initial.cost}--{self.rem_path}"
	
problem1 = ExplicitGraphSearchProblem(edges=
[Edge('A','B',3), Edge('A','C',1), Edge('B','D',1), Edge('B','G',3),
Edge('C','B',1), Edge('C','D',3), Edge('D','G',1)],
start = 'A', goals = {'G'})

problem2 = ExplicitGraphSearchProblem(edges=
[Edge('A', 'B', 1), Edge('A', 'H', 3), Edge('B', 'C', 3), Edge('B', 'D', 1), Edge('D', 'E', 3), Edge('D', 'G', 1), Edge('H', 'J', 1)],
start = 'A', goals = {'G'})

problem3 = ExplicitGraphSearchProblem(edges=
[Edge('A', 'B', 2), Edge('A', 'C', 3), Edge('A', 'D', 4), Edge('B', 'E', 2), Edge('B', 'F', 3), Edge('C', 'J', 7), Edge('D', 'H', 4), Edge('F', 'D', 2), Edge('H', 'G', 3), Edge('J', 'G', 4)],
start='A', goals={'G'})

class Searcher(ABC):
	@abstractmethod
	def search(self):
		pass

class DFSSearcher(Searcher):
	def __init__(self, problem):
		self.problem = problem
		self.path = Path(self.problem.start_node())
		self.visited = set([problem.start_node()])
	
	def search(self):
		if self.problem.is_goal(self.path.peek()[0]):
			yield self.path
		neighbors = self.problem.neighbors(self.path.peek()[0])
		for neighbor in neighbors:
			neighbor, cost = neighbor.v, neighbor.cost
			if neighbor in self.visited:
				continue
			self.visited.add(neighbor)
			self.path.add(neighbor, cost)
			yield from self.search()
			self.path.remove()
			self.visited.remove(neighbor)

class FrontierPQ:
	def __init__(self):
		self.frontier_index = 0
		self.pqueue = []
	
	def push(self, path):
		heapq.heappush(self.pqueue, (path.cost, self.frontier_index, path))
		self.frontier_index += 1
	
	def pop(self):
		return heapq.heappop(self.pqueue)[2]
	
	def is_empty(self):
		return len(self.pqueue) == 0
	
class UniformCostSearcher(Searcher):
	def __init__(self, problem):
		self.problem = problem
		self.visited = set()
		self.frontier = FrontierPQ()
		self.frontier.push(Path(self.problem.start_node()))
	
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
				self.frontier.push(new_path)
		
searcher = DFSSearcher(problem1)


for path in searcher.search():
	print(path)	