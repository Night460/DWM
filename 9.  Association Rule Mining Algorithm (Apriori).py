import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction data
data = {
    'TransactionID': [1, 2, 3, 4, 5],
    'Items': [['Bread', 'Milk'],
              ['Bread', 'Diaper', 'Beer', 'Eggs'],
              ['Milk', 'Diaper', 'Beer', 'Cola'],
              ['Bread', 'Milk', 'Diaper', 'Beer'],
              ['Bread', 'Milk', 'Cola']]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# One-hot encoding of items
oht = df['Items'].str.join('|').str.get_dummies()

# Generate frequent itemsets with a minimum support threshold
frequent_itemsets = apriori(oht, min_support=0.4, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display the frequent itemsets and association rules
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)
