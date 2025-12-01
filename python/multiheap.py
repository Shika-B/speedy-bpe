class MultisetHeap:
    def __init__(self):
        self.d = {}  # dict[key, pos in heap]
        self.heap = []  # list[count, key]

    def add(self, key, count):
        if key in self.d:
            pos = self.d[key]
            (cnt, key) = self.heap[pos]
            self.heap[pos] = (cnt + count, key)
            self._sift_up(pos)
        else:
            self.heap.append((count, key))
            pos = len(self.heap) - 1
            self.d[key] = pos
            self._sift_up(pos)

    def sub(self, key, count):
        pos = self.d[key]
        (cnt, key) = self.heap[pos]
        if cnt > count:
            self.heap[pos] = (cnt - count, key)
            self._sift_down(pos)
        elif cnt == count:
            self.delete(key)
        else:
            raise ValueError("Cannot remove move than the existing count")

    def delete(self, key):
        pos = self.d[key]
        del self.d[key]
        (last_cnt, last_key) = self.heap.pop()
        if len(self.heap) == 0 or pos == len(self.heap):
            return
        self.heap[pos] = (last_cnt, last_key)
        self.d[last_key] = pos
        self._sift_up(pos)
        self._sift_down(pos)

    def popmax(self):
        cnt, key = self.heap[0]
        del self.d[key]

        (last_cnt, last_key) = self.heap.pop()
        if self.heap:
            self.heap[0] = (last_cnt, last_key)
            self.d[last_key] = 0
            self._sift_down(0)

        return (cnt, key)

    def _sift_up(self, pos):
        while pos > 0:
            parent = (pos - 1) // 2
            if self.heap[parent][0] >= self.heap[pos][0]:
                break
            self._heapswap(parent, pos)
            pos = parent

    def _sift_down(self, pos):
        while 2 * pos + 1 < len(self.heap):
            left, right = 2 * pos + 1, 2 * pos + 2
            max_child = left
            if right < len(self.heap) and self.heap[left][0] < self.heap[right][0]:
                max_child = right
            if self.heap[max_child][0] <= self.heap[pos][0]:
                break
            self._heapswap(max_child, pos)
            pos = max_child

    def _heapswap(self, pos1, pos2):
        (_, key1), (_, key2) = self.heap[pos1], self.heap[pos2]
        self.heap[pos1], self.heap[pos2] = self.heap[pos2], self.heap[pos1]
        self.d[key1], self.d[key2] = pos2, pos1

    def __repr__(self):
        return f"{self.d}\n{self.heap}"
