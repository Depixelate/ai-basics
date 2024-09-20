from abc import ABC,abstractmethod
import copy

class NPlayerGame(ABC):
	
	@abstractmethod
	def start_node(self):
		pass
	
	@abstractmethod
	def next_states(self, state, player):
		pass
	
	@abstractmethod
	def getTerminalValue(self, node):
		pass
	
	@abstractmethod
	def next
	
	@abstractmethod
	@property
	def numPlayers(self):
		pass

class Minimax():
	
	def __init__(self, game, start_state):
		self.game = game
		self.start_state = start_state
		self.values = {}
	
	def getValue(self, state, player):
			if state in self.values:
				return self.values[state]
			next_board_states = self.game.next_states(state, player)
			values = [self.getValue(next_state, self.game.next_player(player)) for next_state in next_board_state]
			
	
	def genValues(self):
		self.getValue(self.start_state, 1)
	
	

