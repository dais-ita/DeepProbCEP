from examples.NIPS.generate_data_utils import generate_data
from examples.NIPS.generate_data_utils import write_initiated_ats


def get_last_index(digit, last_digits):
    try:
        return len(last_digits) - last_digits[::-1].index(digit) - 1
    except ValueError:
        return None


def get_digit_initiated_at_200(digit, last_digits, threshold, available_digits):
    if digit in [1, 2, 3, 4, 5, 6, 7, 8]:
        prev_digit = digit - 1

        index = get_last_index(prev_digit, last_digits)

        if index and 9 not in last_digits[index:]:
            # If 9 is not between the last prev_digit and the end of the window, generate the complex event
            return digit, True

    return None, None


if __name__ == '__main__':
    generate_data(
        scenario_function=get_digit_initiated_at_200,
        test_function=get_digit_initiated_at_200,
        relevant_digits=9,
        noises_function=lambda: [0.0],
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True,
        start_sequence=[1, 3, 5, 7],
        end_sequence=[2, 4, 6, 8]
    )
