from pingouin import ttest
x = [53.07, 39.41, 91.27, 90.29]
print(ttest(x, 4).round(2))