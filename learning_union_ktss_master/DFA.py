class DFA():
	"""
		Class for Deterministic finite automaton (DFA)
	"""
	
	
	def __init__(self, initialState="", acceptingStates={}, states={}, transitions={}, predecessor={}, alphabet={}):
		"""
			Constructor for DFA class. Returns a DFA
				initialState : starting state
				acceptingStates : set of accepting states
				states : set of states
				transitions : map transitions[start_state][symbol] = target_state
				predecessor : map predecessor[target_state][symbol] = start_state
				alphabet : alphabet
		"""
		self.initialState = initialState
		self.acceptingStates = acceptingStates
		self.states = states
		self.transitions = transitions
		self.predecessor = predecessor
		self.alphabet = alphabet



	def clean(self):
		"""
			Removes unnecessary transitions, and unreachable states
		"""
		#Get the reached states
		reached = {}
		reached[self.initialState] = None
		reached = self.reachables(self.initialState, reached)
		notreached = []
		#Delete unreached states
		# for s in self.states:
			# if s not in reached:
				# notreached.append(s)
		notreached = list(set(self.states) - set(reached))
		for nr in notreached:
			del self.states[nr]
		#Delete unreachable acceptingStates
		for accepting in list(self.acceptingStates):
			if accepting not in self.states:
				del self.acceptingStates[accepting]
		#Delete useless transitions
		for trans in list(self.transitions):
			for char in list(self.transitions[trans]):
				if self.transitions[trans][char] not in self.states:
					del self.transitions[trans][char]
			if trans not in self.states:
				del self.transitions[trans]
		for trans in list(self.predecessor):
			#print(len(self.predecessor))
			char, prec = self.predecessor[trans]
			if trans not in self.states:
				del self.predecessor[trans]
				continue
			if prec not in self.states:
				for t in self.transitions:
					for c in self.transitions[t]:
						if self.transitions[t][c] == trans:
							self.predecessor[trans] = (c,t)


	def isAccpeted(self,line):
		state = self.initialState
		words = line.split(" ")
		i = 0
		while i < len(words):
			action = words[i]
			if action not in self.transitions[state]:
				return 0
			nextState = self.transitions[state][action]
			if nextState in self.acceptingStates:
				return 1
			else:
				state = nextState
			i += 1

		return 0
        
        
	def reachables(self, state, reached):
		"""
			Returns reachable states from a given starting state
				state = starting state we start to explore the automaton from
		"""
		try:
			for char in self.transitions[state]:
				if self.transitions[state][char] not in reached:
					reached[self.transitions[state][char]] = None
					reached = self.reachables(self.transitions[state][char], reached)
		except KeyError:
			pass
		return reached



	def is_reachable_from(self,state,destination):
		reached = {}
		reached[state] = None
		reached = self.reachables(state, reached)
		if destination in reached:
			return True
		return False



	def toDot(self):
		"""
			Returns a dot representation of the automaton
		"""
		#Header
		dot = "digraph finite_state_machine {\n\trankdir=LR;\n"
		dot += "\tnode [shape=plaintext, label=\"\"]; arrowInit;\n"
		#Accepting states
		dot += "\tnode [shape = doublecircle, label = \"\"]; "
		for accepting in self.acceptingStates:
			if accepting == '':
				dot += "initialState "
			else:
				dot += accepting + " "
		
		dot += ";\n"
		#Rejecting states
		rejectingStates = list(set(self.states) - set(self.acceptingStates))
		if rejectingStates:
			dot += "\tnode [shape = circle, label = \"\"]; "
			for rejecting in rejectingStates:
				if rejecting == '':
					dot += "initialState "
				else:
					dot += rejecting + " "
			dot += ";\n"
		#Transitions
		dot += "\tarrowInit -> initialState [ label=\"\" ];\n"
		for trans in self.transitions:
			for char in self.transitions[trans]:
				if trans == '':
					start = 'initialState'
				else:
					start = trans
				if self.transitions[trans][char] == '':
					to = 'initialState'
				else:
					to = self.transitions[trans][char]
				dot += "\t " + start + " -> " + to + " [ label=\"" + char + "\" ];\n"
		dot += "}"
		return dot


	def accpetanceDistance(self,line):
		state = self.initialState
		words = line.split(" ")
		i = 0
		distance = 0
		while i < len(words):
			action = words[i]
			if action not in self.transitions[state]:
				distance += 1
			else:
				nextState = self.transitions[state][action]
				if nextState in self.acceptingStates:
					return distance/len(words)
				else:
					state = nextState
			i += 1

		return 1
            