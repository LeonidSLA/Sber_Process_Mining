import numpy as np


def aggregate_traces(*vertices: np.ndarray) -> np.ndarray:
    """
    Encodes event traces as a sequence of stage vertex vectors.
    The vector for A->B->C will be different from C -> B -> A.

    Parameters
    ----------
    vertices: Tuple[np.ndarray]
    Sequence of embedded vertex vectors in the trace.

    Returns
    -------
    result: np.ndarray
    Vector corresponding to an event trace.
    """
    assert (vertices is not None), (
        "Send in a sequence of vertex vectors"
    )
    n = int(np.ceil(np.sqrt(vertices[0].shape[0])))
    output = np.identity(n)
    for vertex in vertices:
        output = output @ np.resize(vertex, (n, n))
    output.resize(1, n ** 2)
    # scaling so that all the values are big enough
    scale = (np.max(output) - np.min(output))
    scale = 1 if (scale == 0) else 1/scale
    output *= scale
    return output
