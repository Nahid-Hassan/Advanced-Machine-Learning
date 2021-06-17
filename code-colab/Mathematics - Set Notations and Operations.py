# Consider the set A={1,3,2,0,1,3}. Find ∣A∣.
A = {1,3,2,0,1,3}

# length of |A|
print(len(A)) # 4

# subset
A = {1,3,4,7,10,11}

# =============== Subset ==================== #
# Checking if A is subset of B (vice versa)
# Returns True
# A is subset of B
B = {1, 7, 11}
print(B.issubset(A)) # True

B = {1, 3, 4, 8}
print(B.issubset(A)) # False

B = set()
print(B.issubset(A)) # True

B = {1, 3, 4, 7, 10, 11}
print(B.issubset(A)) # True

# =========== Intersection and Union ================== #
A = {1,2,4,5,6}
B = {1,2,3,5,7}

print(A.intersection(B)) # {1,2,5}
print(A.union(B)) # {1,2,3,4,5,6,7}

# =============== Element  ================ #
# If some object 𝑥 is an element of 𝐴 we write 𝑥 ∈ 𝐴
print (2 in A) # True
print(3 in B) # True
print(3 in A) # False