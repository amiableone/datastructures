import time
from typing import List, Tuple, Optional, Union
from collections.abc import Iterable
from re import sub as re_sub


def timer(func):
    def wrapper(*args, **kwargs):
        s = time.perf_counter()
        res = func(*args, **kwargs)
        e = time.perf_counter() - s
        print(f'Execution time: {e:.6f}')
        return res

    return wrapper


class TreeNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

    def __repr__(self):
        return f"<BinaryTree: {self.to_tuple()}>"

    def __str__(self):
        return f"<BinaryTree: {self.to_tuple()}>"

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return self.key == other.key

        raise TypeError(
            f"Can't compare with an instance of "
            f"{type(other)}"
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, TreeNode):
            return self.key < other.key

        raise TypeError(
            f"Can't compare with an instance of "
            f"{type(other)}"
        )

    def __gt__(self, other):
        if isinstance(other, TreeNode):
            return self.key > other.key

        raise TypeError(
            f"Can't compare with an instance of "
            f"{type(other)}"
        )

    def __le__(self, other):
        if isinstance(other, TreeNode):
            return self.key <= other.key

        raise TypeError(
            f"Can't compare with an instance of "
            f"{type(other)}"
        )

    def __ge__(self, other):
        if isinstance(other, TreeNode):
            return self.key >= other.key

        raise TypeError(
            f"Can't compare with an instance of "
            f"{type(other)}"
        )

    def __len__(self):
        return self.size()

    def __bool__(self):
        if self is None:
            return False

        if self.key is None:
            return any([
                type(self).__bool__(self.left),
                type(self).__bool__(self.right)
            ])

        return True

    def is_bst(self):
        """
        Check whether the tree is a
        Binary Search Tree. That is, if, for any node,
        its key is greater than any key contained in
        its left subtree and lower than any key contained
        in its right subtree.
        """
        if not self:
            return True

        left_comparison = (self.left <= self) if self.left else True
        right_comparison = (self.right >= self) if self.right else True

        return (
            left_comparison and
            right_comparison and
            TreeNode.is_bst(self.left) and
            TreeNode.is_bst(self.right)
        )

    def is_balanced(self):
        """
        Check whether a tree is balanced.
        That is, if for all nodes left and right
        subtrees differ in size by no more than 1.
        """
        res, _ = self._is_balanced()
        return res

    def _is_balanced(self):
        if self is None:
            return True, 0

        balanced_l, height_l = type(self)._is_balanced(self.left)
        balanced_r, height_r = type(self)._is_balanced(self.right)
        similar_heights = abs(height_l - height_r) <= 1

        balanced = balanced_l and balanced_r and similar_heights
        height = 1 + max(height_l, height_r)

        return balanced, height

    def height(self):
        if self is None:
            return 0
        return (1 + max(
            TreeNode.height(self.left),
            TreeNode.height(self.right)
        ))

    def size(self):
        if self is None:
            return 0
        return (
            1 +
            TreeNode.size(self.left) +
            TreeNode.size(self.right)
        )

    @classmethod
    def from_tuple(cls, data):
        """
        Compile a TreeNode from a tree-tuple.
        """
        if data is None:
            node = None

        elif isinstance(data, tuple) and len(data) == 3:
            node = cls(data[1])
            node.left = cls.from_tuple(data[0])
            node.right = cls.from_tuple(data[2])

            if node.left:
                node.left.parent = node

            if node.right:
                node.right.parent = node

        else:
            node = cls(data)

        return node

    def to_tuple(self):
        """
        Return a representation of the instance
        as a tree-tuple.
        """
        if self is None:
            return None

        if self.left is None and self.right is None:
            return (self.key, self.value) if self.value else self.key

        return (
            TreeNode.to_tuple(self.left),
            (self.key, self.value),
            TreeNode.to_tuple(self.right)
        ) if self.value else (
            TreeNode.to_tuple(self.left),
            self.key,
            TreeNode.to_tuple(self.right)
        )

    def display_keys(self, level=0):
        """
        Print out a visualization of the instance,
        albeit rotated 90 degrees counterclockwise.
        """
        space = '\t'
        if self is None:
            print(space*level + '∅')
            return

        if self.left is None and self.right is None:
            print(space*level + str(self.key))
            return

        TreeNode.display_keys(self.right, level+1)
        print(space*level + str(self.key))
        TreeNode.display_keys(self.left, level+1)

    def traverse_inorder(self):
        if self is None:
            return []
        return (
            TreeNode.traverse_inorder(self.left) +
            [self.key] +
            TreeNode.traverse_inorder(self.right)
        )

    def traverse_preorder(self):
        if self is None:
            return []
        return (
            [self.key] +
            TreeNode.traverse_preorder(self.left) +
            TreeNode.traverse_preorder(self.right)
        )

    def traverse_postorder(self):
        if self is None:
            return []
        return (
            TreeNode.traverse_postorder(self.left) +
            TreeNode.traverse_postorder(self.right) +
            [self.key]
        )

    def morris_traversal(self):
        curr = self
        traversed = []
        while curr:
            if curr.left is None:
                traversed.append(curr.key)
                curr = curr.right

            else:
                pre = curr.left
                while pre.right and pre.right != curr:
                    pre = pre.right

                if pre.right is None:
                    pre.right = curr
                    curr = curr.left

                else:
                    pre.right = None
                    traversed.append(curr.key)
                    curr = curr.right

        return traversed

    def bfs(self):
        if self is None:
            return []
        queue = [self]
        idx = 0
        while idx < len(queue):
            curr = queue[idx]
            idx += 1
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)

        return [node.key for node in queue]

    def mirror(self):
        if self is None:
            return
        queue = [self]
        idx = 0
        while idx < len(queue):
            curr = queue[idx]
            idx += 1
            curr.left, curr.right = curr.right, curr.left
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)


class BSTNode(TreeNode):
    def __init__(self, key, value=None, parent=None):
        super().__init__(key, value)
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None
        self.parent: Optional[BSTNode] = parent
        self._size: int = 1
        BSTNode._adjust_size(self, upward=True)

    def __repr__(self):
        return f"<BinarySearchTree: {self.to_tuple()}>"

    def __str__(self):
        return f"<BinarySearchTree: {self.to_tuple()}>"

    def __len__(self):
        if not self:
            return 0

        return self._size

    def size(self):
        """
        This method complexity is O(n).
        Use len(obj) which complexity is O(1).
        """
        if self is None:
            return 0

        self._size = (
            type(self).size(self.left) +
            1 +
            type(self).size(self.right)
        )
        return self._size

    # TODO:
    #   Maybe `is_balanced()` now can be implemented
    #   with better complexity based on relation
    #   between self.height() and self.size() because
    #   self.size() is O(1) now.

    def min_node(self):
        current = self
        while current.left:
            current = current.left
        return current

    def max_node(self):
        current = self
        while current.right:
            current = current.right
        return current

    def find(self, key):
        if self is None:
            raise KeyError(f"Key {key} not found")

        if self.key == key:
            return self

        if self.key > key:
            return type(self).find(self.left, key)

        if self.key < key:
            return type(self).find(self.right, key)

    def insert(self, key, value=None, parent=None):
        """
        Insert the key-value pair in a new node
        or raise KeyError if key already exists.
        Perform insertion in-place if self is not None.

        WARNING: Don't use with non-root node.
        """
        if self is None:
            try:
                return parent.__class__(key, value, parent)
            except TypeError:
                raise TypeError("Insertion to None is not"
                                "supported.")

        if self.key > key:
            self.left = type(self).insert(
                self.left, key, value, self
            )

        elif self.key < key:
            self.right = type(self).insert(
                self.right, key, value, self
            )

        else:
            raise KeyError(
                f"Key {key} already exists."
            )

        return self

    def update(self, key, value=None):
        """
        Find a node which :attr:key equals key
        and update the value of its :attr:value
        with value. If not found, raise KeyError.
        """
        node = self.find(key)
        node.value = value

    def delete(self, key):
        if self is None:
            raise KeyError(f"Key {key} not found")

        if self.key == key:
            self._fill_deleted_node()

        elif self.key > key:
            type(self).delete(self.left, key)

        else:
            type(self).delete(self.right, key)

    def _fill_deleted_node(self):
        """
        Replace a node with the rightmost node
        of its left subtree or the leftmost node
        of its right subtree, whichever is the longest.
        If the node is a tree root with no children,
        set its key and value attributes to None.
        The operation is performed in-place.
        """
        # Choose a filler node from the longest
        # subtree, if any.
        length_l = type(self).__len__(self.left)
        length_r = type(self).__len__(self.right)

        if length_l == length_r == 0:
            if not self.parent:
                self.key, self.value = None, None
            elif self.parent.left is self:
                self._adjust_size()
                self.parent.left = None
            else:
                self._adjust_size()
                self.parent.right = None

        elif length_l >= length_r:
            self._replace_with_rightmost_node_of_left_subtree()

        else:
            self._replace_with_leftmost_node_of_right_subtree()

    def _replace_with_rightmost_node_of_left_subtree(self):
        """
        Update the key-value pair of a node with that
        of its left subtree rightmost node.
        """
        filler = self.left
        if filler:
            while filler.right:
                filler = filler.right

        self.key, self.value = filler.key, filler.value
        filler._adjust_size()

        # Replace the node filler points to with its
        # left child. Question is whether it's left or
        # right child of its parent.
        if filler.parent is self:
            filler.parent.left = filler.left
        else:
            filler.parent.right = filler.left
        if filler.left:
            filler.left.parent = filler.parent

    def _replace_with_leftmost_node_of_right_subtree(self):
        """
        Update the key-value pair of a node with that
        of its right subtree leftmost node.
        """
        filler = self.right
        if filler:
            while filler.left:
                filler = filler.left

        self.key, self.value = filler.key, filler.value
        filler._adjust_size()

        if filler.parent is self:
            filler.parent.right = filler.right
        else:
            filler.parent.left = filler.right
        if filler.right:
            filler.right.parent = filler.parent

    def _adjust_size(self, upward=False):
        """
        Adjust lengths of all ascendants of a node
        up to the root by 1.
        """
        if self is None:
            return
        if self.parent is None:
            return
        adj = 1 if upward else -1
        parent = self.parent
        parent._size += adj
        while parent.parent:
            parent = parent.parent
            parent._size += adj

    def balance(self):
        """
        Balance the tree (not in-place).
        This method converts the tree to BST, too.
        """
        data = self.to_list()
        if self.is_bst():
            return type(self)._from_sorted(data)

        return type(self).from_seq(data)

    def to_list(self):
        """
        Return a list of key-value pairs
        of a Binary Search Tree by traversing
        the tree in-order.
        """
        if self is None:
            return []

        return (
            type(self).to_list(self.left) +
            [(self.key, self.value)] +
            type(self).to_list(self.right)
        )

    def _is_skewed(self):
        """
        Check if a tree is skewed and return
        a height difference of its subtrees.
        Negative height difference means the tree
        is skewed to the left and vice-versa.
        """
        if self is None:
            return False, 0

        balanced_l, height_l = type(self)._is_balanced(self.left)
        balanced_r, height_r = type(self)._is_balanced(self.right)
        height_diff = height_l - height_r

        skewed = not balanced_l or not balanced_r or abs(height_diff) > 1
        return skewed, height_diff

    @classmethod
    def from_seq(cls, data):
        if not isinstance(data, Iterable):
            return

        data = sorted(data)
        return cls._from_sorted(data)

    @classmethod
    def _from_sorted(cls, data, lo=0, hi=None, parent=None):
        """
        Create a Binary Tree from a sequence.
        For the tree to be a Binary Search Tree,
        data HAS TO BE sorted.

        Using unsorted data will return a BSTNode
        instance with methods not working correctly.
        """
        if data is None:
            return

        if hi is None:
            hi = len(data) - 1

        if lo > hi:
            return

        mid = (lo + hi) // 2
        try:
            key, value = data[mid]
        except TypeError:
            key, value = data[mid], None

        node = cls(key, value)
        node.parent = parent
        node._size = hi - lo + 1
        node.left = cls._from_sorted(data, lo, mid - 1, node)
        node.right = cls._from_sorted(data, mid + 1, hi, node)

        return node

    @classmethod
    def from_tuple(cls, data, parent=None):
        if data is None:
            node = None

        elif isinstance(data, tuple) and len(data) == 3:
            try:
                key, value = data[1]
            except TypeError:
                key, value = data[1], None

            node = cls(key, value, parent=parent)
            node.left = cls.from_tuple(data[0], node)
            node.right = cls.from_tuple(data[2], node)

        else:
            try:
                key, value = data
            except TypeError:
                key, value = data, None

            node = cls(key, value, parent=parent)

        return node

    def to_bst(self, bst=None, inplace=False):
        """
        Convert a tree into a BST without balancing
        like `balance()` does.
        The non-inplace algorithm puts keeps the root
        node of the original tree the root node of the
        resulting tree. Hence, different structure of
        the new tree.
        """
        if self is None:
            return bst

        if inplace:
            data_list = sorted(self.to_list())
            curr = self
            while curr:
                if curr.left:
                    pre = curr.left
                    while pre.right and pre.right != curr:
                        pre = pre.right
                    if not pre.right:
                        pre.right = curr
                        curr = curr.left
                    else:
                        data = data_list.pop(0)
                        curr.key, curr.value = data
                        pre.right = None
                        curr = curr.right
                else:
                    data = data_list.pop(0)
                    curr.key, curr.value = data
                    curr = curr.right

        else:
            bst = type(self).insert(bst, self.key, self.value)
            bst = type(self).to_bst(self.left, bst, False)
            bst = type(self).to_bst(self.right, bst, False)
            return bst

    def paths(self):
        paths = []
        path = ""
        curr = self
        while curr:
            if curr.left:
                pre = curr.left
                while pre.right and pre.right != curr:
                    pre = pre.right
                if not pre.right:
                    pre.right = curr
                    curr = curr.left
                    path += "0"
                else:
                    paths.append(path + "-")
                    if curr.right.parent is curr:
                        path += "1"
                    else:
                        # curr.right is curr.parent or
                        # curr is the rightmost node of
                        # the left subtree of curr.right
                        path = re_sub(r"01*$", "", path)
                    pre.right = None
                    curr = curr.right
            else:
                paths.append(path + "-")
                if not curr.right or curr.right.parent is not curr:
                    path = re_sub(r"01*$", "", path)
                else:
                    path += "1"
                curr = curr.right
        return paths

    @staticmethod
    def _tree_tuple_to_list(data):
        if not isinstance(data, tuple):
            return []

        if isinstance(data, tuple) and len(data) == 3:
            l, m, r = data
            return (
                BSTNode._tree_tuple_to_list(l) +
                [m] +
                BSTNode._tree_tuple_to_list(r)
            ) if isinstance(m, tuple) and len(m) == 2 else (
                BSTNode._tree_tuple_to_list(l) +
                [(m, m)] +
                BSTNode._tree_tuple_to_list(r)
            )


class AVLNode(BSTNode):
    def __init__(self, key, value=None, parent=None):
        super().__init__(key, value, parent)
        self.left: Optional[AVLNode] = None
        self.right: Optional[AVLNode] = None
        self.parent: Optional[AVLNode] = parent

    def _rotate_left(self):
        """
        Node0[None, Node1[None, Node2]] ->
        Node1[Node0, Node2].
        """
        root = self.right
        root.parent = self.parent
        self.right = None
        root.left = self
        root.left.parent = root
        root.size()
        return root

    def _rotate_right(self):
        """
        Node2[Node1[Node0, None], None] ->
        Node1[Node0, Node2].
        """
        root = self.left
        root.parent = self.parent
        self.left = None
        root.right = self
        root.right.parent = root
        root.size()
        return root

    def _rotate_left_right(self):
        """
        Node2[Node0[None, Node1], None] ->
        Node1[Node0, Node2]
        """
        self.left: AVLNode
        self.left = self.left._rotate_left()
        return self._rotate_right()

    def _rotate_right_left(self):
        """
        Node0[None, Node2[Node1, None]] ->
        Node1[Node0, Node2]
        """
        self.right: AVLNode
        self.right = self.right._rotate_right()
        return self._rotate_left()

    def _rotate_single_child_node(self):
        size_diff = self._size_diff()
        if size_diff >= 2:
            size_diff_l = self.left._size_diff()
            if size_diff_l == 1:
                return self._rotate_right()
            elif size_diff_l == -1:
                return self._rotate_left_right()
        elif size_diff <= -2:
            size_diff_r = self.right._size_diff()
            if size_diff_r == 1:
                return self._rotate_right_left()
            elif size_diff_r == -1:
                return self._rotate_left()
        return

    def _size_diff(self):
        if self is None:
            return 0

        return (
            type(self).__len__(self.left) -
            type(self).__len__(self.right)
        )

    # def insert(self, key, value=None, parent=None):
    #     super().insert(key, value, parent)
    #     node = self.find(key)
    #     node._rotate_after_insert()
    #
    # def _rotate_after_insert(self):
    #     """Run on an inserted node"""
    #     if not self.parent:
    #         return
    #
    #     if not self.parent.parent:
    #         return
    #
    #     ggp = self.parent.parent.parent
    #     if not ggp:
    #         root = self.parent.parent._rotate_single_child_node()
    #         self.key, self.value, self.parent =\
    #             root.key, root.value, root.parent
    #     elif ggp.left is self.parent.parent:
    #         ggp.left = ggp.left._rotate_single_child_node()
    #     else:
    #         ggp.right = ggp.right._rotate_single_child_node()


class TreeMap:
    def __init__(self, tree_type=AVLNode):
        self.root: Optional[tree_type] = None
        self._root_type = tree_type
        self._inserted_times: int = 0

    def __setitem__(self, key, value):
        try:
            self.root.update(key, value)
        except KeyError:
            self.root.insert(key, value)
            if self._inserted_times >= 1000:
                self.root.balance()
                self._inserted_times = 0
            else:
                self._inserted_times += 1
        except AttributeError:
            self.root = self._root_type(key, value)

    def __getitem__(self, key):
        try:
            node = self.root.find(key)
            return node.value

        except (KeyError, AttributeError):
            raise KeyError(
                f"Key {key} not found"
            )

    def __delitem__(self, key):
        try:
            self.root.delete(key)
        except KeyError:
            return

    def __len__(self):
        return BSTNode.__len__(self.root)

    def __iter__(self):
        return (n for n in BSTNode.to_list(self.root))

    def __contains__(self, key):
        try:
            self.root.find(key)
            return True
        except (KeyError, AttributeError):
            return False

    def __repr__(self):
        return BSTNode.__repr__(self.root)

    def __str__(self):
        return BSTNode.__str__(self.root)

    def display(self):
        return BSTNode.display_keys(self.root)


class Graph:
    def __init__(
            self,
            num_nodes: int,
            edges: List[Tuple],
            directed=False
    ):
        self.num_nodes = num_nodes
        self.data = [[] for _ in range(num_nodes)]
        self.directed = directed
        self.weighted = len(edges) > 0 and len(edges[0]) == 3
        if self.weighted:
            edges: List[Tuple[int, int, int]]
            for n1, n2, w in edges:
                self.data[n1].append((n2, w))
                if not self.directed:
                    self.data[n2].append((n1, w))
        else:
            edges: List[Tuple[int, int]]
            for n1, n2 in edges:
                self.data[n1].append(n2)
                if not self.directed:
                    self.data[n2].append(n1)
        self._display = None

    def __repr__(self):
        if self._display is not None:
            return self._display

        res = ''
        for i, nodes in enumerate(self.data):
            res += f'{i}: {nodes}\n'

        self._display = res
        return res

    def __str__(self):
        return self.__repr__()

    def bfs(self, node=0):
        """
        Traverse through all the nodes
        starting from `node`.
        """
        traversed = []
        discovered = [False for _ in self.data]
        discovered[node] = True
        queue = [node]

        if self.weighted:
            self.data: List[List[Tuple]]

            while queue:
                curr = queue.pop(0)
                traversed.append(curr)

                for node, _ in self.data[curr]:
                    if not discovered[node]:
                        discovered[node] = True
                        queue.append(node)

        else:
            self.data: List[List[int]]

            while queue:
                curr = queue.pop(0)
                traversed.append(curr)

                for node in self.data[curr]:
                    if not discovered[node]:
                        discovered[node] = True
                        queue.append(node)

        return traversed

    def dfs(self, node=0):
        """
        Traverse through all the nodes
        starting from `node`.
        """
        traversed = []
        discovered = [False for _ in self.data]
        discovered[node] = True
        stack = [node]

        if self.weighted:
            self.data: List[List[Tuple]]

            while stack:
                curr = stack.pop()
                traversed.append(curr)

                for node, _ in self.data[curr]:
                    if not discovered[node]:
                        discovered[node] = True
                        stack.append(node)

        else:
            self.data: List[List[int]]

            while stack:
                curr = stack.pop()
                traversed.append(curr)

                for node in self.data[curr]:
                    if not discovered[node]:
                        discovered[node] = True
                        stack.append(node)

        return traversed

    def shortest_path(
            self,
            start,
            end,
            memo: Optional[List] = None
    ):
        """
        Find the shortest path starting in node
        `start` and finishing in node `end`.
        """
        if not memo:
            memo = [False] * self.num_nodes

        if start == end:
            return 0
        elif memo[start]:
            return None

        memo[start] = True
        distances = []

        if self.weighted:
            for node, weight in self.data[start]:
                rest = self.shortest_path(
                    node, end, memo.copy()
                )
                if rest is not None:
                    distances.append(weight + rest)

        elif end in self.data[start]:
            return 1

        else:
            for node in self.data[start]:
                rest = self.shortest_path(
                    node, end, memo.copy()
                )
                if rest is not None:
                    distances.append(1 + rest)

        return min(distances) if distances else None
