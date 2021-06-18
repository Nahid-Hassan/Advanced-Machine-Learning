# Tuples

# Number of Passwords
# How many different 5-symbol passwords can we create using lower case Latin letters only? (the size of the alphabet is 26)

def no_of_passwords(k=0):
    """
        All are lowercase english alphabet. So for each position we have 26 possibilities.  
    
        length_of_passwords = 5
        each_position_no_of_possibilities = 26
    """
    n = 26
    k = 5

    return n**k

def cartesian_products(A, B):
    cross_product = []

    for x in A:
        for y in B:
            cross_product.append((x, y))
    
    return cross_product
 
def number_with_exactly_one_7_digits(n = 0):
    import math
    return math.factorial(n) 

if __name__ == "__main__":
    print(no_of_passwords(k=5)) # 11881376
    print(cartesian_products(A={1,2,3}, B={2,3,4}))
    # [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)]
    print(number_with_exactly_one_7_digits(n=10))# 3628800