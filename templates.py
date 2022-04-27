"""
Suffix Automaton

Examples:
1923
"""
class State:
    def __init__(self, length, link, next):
        self.length = length
        self.link = link
        self.next = next
        self.max = 0
        self.ans = length

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


"""
KMP

Examples:
"""
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

# s在t中的所有出现
def find_appearance(s, t):
    kmp = prefix_function(s + '#' + t)
    res = []
    for i in range(len(s) + 1, len(s) + 1 + len(t)):
        if kmp[i] == len(s):
            res.append(i - 2 * len(s)) # t[i - 2 * len(s) : i - len(s) - 1]
    return res

# print(find_appearance('cdc', 'cdcdcdcdcdc'))
# [0, 2, 4, 6, 8]


"""
Manacher

Examples:
5, 131, 132, 1960
"""
def manacher(s):
    # d1: 最长奇数回文半径
    d1 = []
    l, r = 0, -1
    for i, c in enumerate(s):
        if i > r:
            k = 1
        else:
            k = min(d1[l + r - i], r - i + 1)
        while i - k >= 0 and i + k < len(s) and s[i - k] == s[i + k]:
            k += 1
        d1.append(k)
        k -= 1
        if i + k > r:
            r = i + k
            l = i - k
    # d2: 最长偶数回文半径
    # 以id为右中心
    d2 = []
    l, r = 0, -1
    for i, c in enumerate(s):
        if i > r:
            k = 0
        else:
            k = min(d2[l + r - i + 1], r - i + 1)
        while i - k - 1 >= 0 and i + k < len(s) and s[i - k - 1] == s[i + k]:
            k += 1
        d2.append(k)
        k -= 1
        if i + k > r:
            r = i + k
            l = i - k - 1
    return d1, d2

# print(manacher('aaaab'))
# ([1, 2, 2, 1, 1], [0, 1, 2, 1, 0])

def max_product(s):
    # "ababbb"
    d1, _ = manacher(s)
    # d1 - 1, 2, 2, 1, 2, 1
    l1 = [0] * len(s)
    l2 = [0] * len(s)
    for i in range(len(s)):
        l1[i + d1[i] - 1] = max(l1[i + d1[i] - 1], d1[i] * 2 - 1)
        l2[i - d1[i] + 1] = max(l2[i - d1[i] + 1], d1[i] * 2 - 1)
    # l1 - 1, 0, 3, 3, 0, 3
    # l2 - 3, 3, 0, 3, 0, 1
    for i in range(1, len(l1)):
        l1[i] = max(l1[i - 1], l1[i])
    for i in range(len(l2) - 2, -1, -1):
        l2[i] = max(l2[i], l2[i + 1])
    # IMPORTANT technique! - e.g. l1[4] = max(l1[4], l1[5] - 2)
    # l1[5] - 2 --> "bbb" to "b"
    for i in range(len(l1) - 2, -1, -1):
        l1[i] = max(l1[i], l1[i + 1] - 2)
    for i in range(1, len(l2)):
        l2[i] = max(l2[i - 1] - 2, l2[i])
    # print(l1)
    # print(l2)
    res = 0
    for i in range(len(s) - 1):
        res = max(res, l1[i] * l2[i + 1])
    return res

# print(max_product("ababbb"))
# 9
# "aba", "bbb"


"""
Z function
z[i]是s和s[i:-1]的最长公共前缀(LCP)的长度

Examples:
2223
"""
def z_function(s):
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r and z[i - l] < r - i + 1:
            z[i] = z[i - l]
        else:
            z[i] = max(0, r - i + 1)
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1
    return z

# print(z_function("aaaaa"))
# [0, 4, 3, 2, 1]
# print(z_function("aaabaab"))
# [0, 2, 1, 0, 2, 1, 0]
# print(z_function("abcabc"))
# [0, 0, 0, 3, 0, 0]


"""
Fenwick Tree

Examples:
"""
def get_sum(bit, i):
    s = 0
    i += 1
    while i > 0:
        s += bit[i]
        i -= i & -i
    return s

def update(bit, n, i, v):
    i += 1
    while i <= n:
        bit[i] += v
        i += i & -i

def construct(arr, n):
    bit = [0] * (n + 1)
    for i in range(n):
        update(bit, n, i, arr[i])
    return bit

# print(construct([1, 1, 4, 2, 5, 2], 6))
# [0, 1, 2, 4, 8, 5, 7]


"""
Union Find

Examples:
1202
"""
n = 10
parent = [i for i in range(n)]
rank = [0] * n
def find(i):
    if parent[i] != i:
        parent[i] = find(parent[i])
    return parent[i]
def union(i, j):
    ri, rj = find(i), find(j)
    if ri != rj:
        if rank[ri] > rank[rj]:
            parent[rj] = ri
        elif rank[ri] < rank[rj]:
            parent[ri] = rj
        else:
            parent[rj] = ri
            rank[ri] += 1
        return True
    return False