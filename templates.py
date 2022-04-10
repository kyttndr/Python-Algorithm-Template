class SAM:
    def __init__(self, path):
        self.states = []
        self.states.append(State(0, - 1, {}))
        self.last = 0
        for c in path:
            self.sam_extend(c)

    def sam_extend(self, c):
        curr = len(self.states)
        self.states.append(State(self.states[self.last].length + 1, -1, {}))
        p = self.last
        while p != -1 and c not in self.states[p].next:
            self.states[p].next[c] = curr
            p = self.states[p].link
        if p == -1:
            self.states[curr].link = 0
        else:
            q = self.states[p].next[c]
            if self.states[q].length == self.states[p].length + 1:
                self.states[curr].link = q
            else:
                clone = len(self.states)
                self.states.append(State(self.states[p].length + 1, self.states[q].link, copy.deepcopy(self.states[q].next)))
                while p != -1 and c in self.states[p].next and self.states[p].next[c] == q:
                    self.states[p].next[c] = clone
                    p = self.states[p].link
                self.states[q].link = clone
                self.states[curr].link = clone
        self.last = curr

    def compare(self, path):
        for state in self.states:
            state.max = 0
        v = l = 0
        for c in path:
            while v and c not in self.states[v].next:
                v = self.states[v].link
                l = self.states[v].length
            if c in self.states[v].next:
                l += 1
                v = self.states[v].next[c]
            self.states[v].max = max(self.states[v].max, l)
        for state in self.states:
            state.ans = min(state.ans, state.max)

    def LCS(self, paths):
        for path in paths:
            self.compare(path)
        return max(map(lambda x : x.ans, self.states))