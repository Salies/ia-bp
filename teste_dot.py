def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

a = [[1,2,3,4,5,6]]
b = [
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5]
]
print(dot_product(a, b))  # 32