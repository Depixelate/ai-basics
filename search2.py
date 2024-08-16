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
			return self.initial
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

problem1 = ExplicitGraphSearchProblem(
edges=add_reverse([Edge('A', 'B', 9), Edge('A', 'C', 4), Edge('A', 'D', 7), Edge('B', 'E', 11), Edge('C', 'E', 17), Edge('C', 'F', 12), Edge('D', 'F', 14), Edge('E', 'Z', 5), Edge('F', 'Z', 9)]),
start='A',
goals={'Z'},
heuristicDict={'A':21,'B':14,'C':18,'D':18,'E':5,'F':8,'Z':0}
)

problem2 = ExplicitGraphSearchProblem(
edges=add_reverse([Edge('A', 'B', 10), Edge('A', 'C', 12), Edge('A', 'D', 5), Edge('B', 'E', 11), Edge('C', 'D', 6), Edge('C', 'E', 11), Edge('C', 'F', 8), Edge('D', 'F', 14)]),
start='A',
goals={'F'},
heuristicDict={'A':10,'B':15,'C':5,'D':5,'E':10,'F':0}
)

problem3 = ExplicitGraphSearchProblem(edges=
[Edge('A', 'B', 2), Edge('A', 'C', 3), Edge('A', 'D', 4), Edge('B', 'E', 2), Edge('B', 'F', 3), Edge('C', 'J', 7), Edge('D', 'H', 4), Edge('F', 'D', 2), Edge('H', 'G', 3), Edge('J', 'G', 4)],
start='A', goals={'G'},
heuristicDict={'A':7,'B':5,'C':9,'D':6,'E':3,'F':5,'G':0,'H':3,'J':4})

astar_searcher = AStarSearcher(problem3)

print("A Star: ")
for path in astar_searcher.search():
	print(path)	

