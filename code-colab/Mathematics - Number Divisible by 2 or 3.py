# Number Divisible by 2
# How many numbers from 1 to 20, inclusive, are divisible by 2?
# How many numbers from 1 to 20, inclusive, are divisible by 3?
# How many numbers from 1 to 20, inclusive, are divisible by 2 or by 3?

divisible_by_2 = 0
divisible_by_3 = 0
divisible_by_2_or_3 = 0

for i in range(1,21):
    if i % 2 == 0:
        divisible_by_2 += 1
    
    if i % 3 == 0:
        divisible_by_3 += 1
    
    if i % 2 == 0 or i % 3 == 0:
        divisible_by_2_or_3 += 1

print(divisible_by_2) # 10 
print(divisible_by_3) # 6 
print(divisible_by_2_or_3) # 13

