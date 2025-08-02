def one_hot_encode(value, categories):
    one_hot = [0] * len(categories)
    if value in categories:
        one_hot[categories.index(value)] = 1
    return one_hot
