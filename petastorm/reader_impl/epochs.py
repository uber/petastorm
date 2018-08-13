import random


def epoch_generator(items, num_epochs, shuffle):
    current_index = 0
    epochs_left = num_epochs

    while (epochs_left is None or epochs_left > 0) and items:

        if current_index == 0 and shuffle:
            random.shuffle(items)

        yield items[current_index]

        current_index += 1

        if current_index >= len(items):
            current_index = 0
            # If iterations was set to None, that means we will iterate until stop is called
            if epochs_left is not None:
                epochs_left -= 1