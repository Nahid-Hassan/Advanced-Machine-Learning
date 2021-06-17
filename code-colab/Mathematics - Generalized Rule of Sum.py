##  Suppose we have 40 videos in our dataset. Each video falls in at least one of the two categories, comedy videos and music videos. It is known that there are 27 comedy videos  and 22 music videos in the dataset. How many videos fall into both categories?

# Rule of Sum
# If there are finite sets 𝐴 and 𝐵, then |𝐴 ∪ 𝐵| = |𝐴| + |𝐵| − |𝐴 ∩ 𝐵|

# Here,
# |A U B| = ?
# |A| = 27
# |B| = 22

# According to general rule of sum,

fall_into_both_categories = 27 + 22 - 40
print(fall_into_both_categories) # 9

## How many integer numbers from 1 to 1000, inclusive, are divisible by 2 or by 3?

divisible_by_2_or_3 = len([x for x in range(1, 1001) if x % 2 == 0 or x % 3 == 0])
print(divisible_by_2_or_3) # 667

## How many integer numbers from 1 to 1000, inclusive, are not divisible neither by 2, nor by 3?

divisible_by_2_or_3 = len([x for x in range(1, 1001) if x % 2 != 0 and x % 3 != 0])
print(divisible_by_2_or_3) # 333
