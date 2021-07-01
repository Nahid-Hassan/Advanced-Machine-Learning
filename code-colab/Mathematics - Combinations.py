import itertools
import math

# Ranking problem

"""
Suppose we have "𝑛" texts and we need to rank
them by relevance. One of the standard
approaches requires to compare each text with
each other. How many comparisons we need to
make?
"""


def ranking(n: int) -> int:
    """
        Compare each text with each other.
        Example: Suppose we have 3 text. 'a', 'b', and 'c'.
        Our solution gives us 3 * 2 = 6 comparisons, because we compare
        each text to other. But if we further review our solutions we see we count same or doing same comparison twice!

        HOW WE SOLVE!!!

        Solve: If we counted each objects k times, just divide the result k times.
    """
    return n * (n - 1) / 2


def combination(n: str, k: int) -> int:
    """
        Return the combination of (n, k) -> 'n choose k'
    """
    # return math.factorial(n) / (math.factorial(k) * math.math.factorial(n - k))
    return list(itertools.combinations(n, k))


def factorial(n):
    return math.factorial(n)

def road_trip(friends: list, k: int):
    """
        friends: str, list or iterables
        k: int, choose number
        return combination of len(friends) and k or "n choose k"
    """
    return list(itertools.combinations(friends, k))

def main():
    print(ranking(10))
    out = combination('abcde', 3)
    print(len(out))
    print(out)

    friends = ['mahin', 'bijoy', 'mony', 'nahid', 'hassan']
    k = 3
    print(road_trip(friends, k))

if __name__ == "__main__":
    main()
