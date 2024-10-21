import numpy as np

def pagerank(links, num_iterations=100, d=0.85):
    num_pages = len(links)
    # Initialize page ranks to 1/n 
    ranks = np.ones(num_pages) / num_pages

    for _ in range(num_iterations):
        new_ranks = np.zeros(num_pages)
        for i in range(num_pages):
            for j in range(num_pages):
                if links[j][i] == 1:  # If there's a link from j to i 
                    new_ranks[i] += ranks[j] / np.sum(links[j])  # Distribute rank 
        ranks = (1 - d) / num_pages + d * new_ranks  # Apply damping factor 

    return ranks

# Example link structure (adjacency matrix)P
# 0 -> 1, 1 -> 2, 2 -> 0, 2 -> 1 
links = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 1, 0]])
ranks = pagerank(links)
print("PageRank Scores:", ranks) 