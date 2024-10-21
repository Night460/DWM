import numpy as np

def hits(links, num_iterations=100):
    num_pages = len(links)

    # Initialize authority and hub scores to 1 for all pages
    authority_scores = np.ones(num_pages)
    hub_scores = np.ones(num_pages)

    # Iterate to update scores
    for _ in range(num_iterations):
        # Update authority scores: A = L^T * H
        authority_scores = np.dot(links.T, hub_scores)
        # Normalize authority scores
        authority_scores /= np.linalg.norm(authority_scores, 2)

        # Update hub scores: H = L * A
        hub_scores = np.dot(links, authority_scores)
        # Normalize hub scores
        hub_scores /= np.linalg.norm(hub_scores, 2)

    return authority_scores, hub_scores


# Example link structure (adjacency matrix)
links = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0]
])

# Run HITS algorithm
authority, hub = hits(links)

# Output the results
print("Authority Scores:", authority)
print("Hub Scores:", hub)
