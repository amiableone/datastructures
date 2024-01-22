from ds import BSTNode, TreeMap, TreeNode
from random import randint
from typing import Optional
from time import perf_counter

tests = []


def create_test(
        name: str,
        treemap: TreeMap,
        deleted_value: Optional[int],
        output: Optional[TreeMap]
):
    test = {
        "name": name,
        "inputs": {
            "map": treemap,
            "del": deleted_value
        },
        "output": output
    }
    global tests
    if test not in tests:
        tests.append(test)
    else:
        print("Test already in Tests")


data_for_tests = []

# Creating data for Test 1.
name1 = "Deleting the only node"

treemap1 = TreeMap()
treemap1.root = BSTNode(15, "15")

output1 = TreeMap()
output1.root = BSTNode(None, None)

data_for_tests.append((name1, treemap1, 15, output1))

# Creating data for Test 2.
name2 = "Deleting the root node with a tree of height 1 as its left child and with no right child"

treemap2 = TreeMap()
treemap2.root = BSTNode(15, "15")
treemap2.root.insert(16, "16")

output2 = TreeMap()
output2.root = BSTNode(16, "16")

data_for_tests.append((name2, treemap2, 15, output2))

# Creating data for Test 3.
name3 = "Deleting the root node with a tree of height 1 as its right child and with no left child"

treemap3 = TreeMap()
treemap3.root = BSTNode(15, "15")
treemap3.root.insert(16, "16")

output3 = TreeMap()
output3.root = BSTNode(16, "16")

data_for_tests.append((name3, treemap3, 15, output3))

# Creating data for Test 4.
name4 = "Deleting the root node of a balanced tree of height 5"

treemap4 = TreeMap()
sequence = [(i, str(i)) for i in range(25)]
treemap4.root = BSTNode.from_seq(sequence)

output4 = TreeMap()
for i in [11, 5, 2, 0, 1, 3, 4, 8, 6, 7, 10, 9, 11, 18, 15, 13, 14, 16, 17, 21, 19, 20, 23, 22, 24]:
    output4[i] = str(i)

data_for_tests.append((name4, treemap4, 12, output4))

# Creating data for Test 5.
name5 = "Deleting the root node of a skewed tree of height >= 5"

treemap5 = TreeMap()
for i in [16, 8, 1, 0, 7, 2, 4, 3, 6, 5, 13, 11, 10, 9, 12, 15, 14, 18, 17, 19, 23, 21, 20, 22, 24]:
    treemap5[i] = str(i)

output5 = TreeMap()
for i in [15, 8, 1, 0, 7, 2, 4, 3, 6, 5, 13, 11, 10, 9, 12, 14, 18, 17, 19, 23, 21, 20, 22, 24]:
    output5[i] = str(i)

data_for_tests.append((name5, treemap5, 16, output5))

# Creating data for Test 6.
name6 = "Deleting the root node in a tree of height >= 5 iteratively x4"

treemap6 = TreeMap()
treemap6.root = BSTNode.from_seq(sequence)

output6 = TreeMap()
for i in [14, 5, 2, 0, 1, 3, 4, 8, 6, 7, 9, 18, 15, 16, 17, 21, 19, 20, 23, 22, 24]:
    output6[i] = str(i)

data_for_tests.append((name6, treemap6, (12, 10, 13, 11), output6))

# Creating data for Test 7.
name7 = "Deleting a node with no children in a tree of height >= 5 iteratively x3"

treemap7 = TreeMap()
treemap7.root = BSTNode.from_seq(sequence)

output7 = TreeMap()
for i in [12, 5, 2, 0, 1, 3, 4, 8, 6, 7, 18, 15, 13, 14, 16, 17, 21, 19, 20, 23, 22, 24]:
    output7[i] = str(i)

data_for_tests.append((name7, treemap7, (9, 11, 10), output7))

# Creating data for Test 8.
name8 = "Deleting all nodes from a balanced tree"

treemap8 = TreeMap()
treemap8.root = BSTNode.from_seq(sequence)

output8 = TreeMap()
output8.root = BSTNode(None, None)

data_for_tests.append((name8, treemap8, "all", output8))

# Creating data for Test 9.
name9 = "Deleting all nodes from a random tree"

treemap9 = TreeMap()
for _ in range(25):
    new = randint(0, 30)
    treemap9[new] = str(new)

output9 = TreeMap()
output9.root = BSTNode(None, None)

data_for_tests.append((name9, treemap9, "all", output9))

# Creating data for Test 10.
name10 = "Deleting a node with a single child iteratively x3"

treemap10 = TreeMap()
for i in range(5):
    treemap10[i] = str(i)

output10 = TreeMap()
output10.root = BSTNode(0, "0")
output10.root.right = BSTNode(4, "4")

data_for_tests.append((name10, treemap10, (3, 2, 1), output10))


# Adding all created tests to `tests`
for test_data in data_for_tests:
    create_test(*test_data)


def eval_test(name, inputs, output):
    treemap, del_val = TreeMap(), inputs["del"]
    treemap.root = TreeNode.from_tuple(inputs["map"].to_tuple())

    if isinstance(del_val, int):
        treemap.root.delete(del_val)
    elif isinstance(del_val, tuple):
        for i in del_val:
            treemap.root.delete(i)
    else:
        # del_val == "all"
        kvlist = list(treemap)
        for k, v in kvlist:
            treemap.root.delete(k)

    return trees_eq(treemap.root, output.root, name)


def trees_eq(left, right, name, level=0):
    if left is None and right is None:
        return True

    if left is None or right is None:
        return False

    res = (
        trees_eq(left.left, right.left, name, level+1) and
        left.key == right.key and
        left.parent == right.parent and
        len(left) == len(right) and
        trees_eq(left.right, right.right, name, level+1)
    )

    if not res and level == 0:
        print(name)
        print("   " * level, "Left ------------------------------")
        left.display_keys()
        print("   " * level, "Right -----------------------------")
        right.display_keys()
        print("\n")

    return res


def eval_all_tests():
    for i, test_data in enumerate(tests):
        s = perf_counter()
        res = eval_test(**test_data)
        e = perf_counter() - s
        print(f"Test {i}, {e / 10 ** 6:.4f} μs -> {res}")


eval_all_tests()
