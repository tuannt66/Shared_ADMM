"""RRT-Connect motion planner in joint space (C-space).

Bidirectional RRT-Connect (Kuffner & LaValle 2000) over an N-DOF joint
configuration space with box joint limits.  Collision checking is
delegated to a user-supplied ``collision_fn(q) -> bool`` (True = the
configuration is in collision), so the planner itself has no PyBullet
dependency and can be unit-tested with synthetic obstacles.

Typical use::

    planner = RRTConnectJointPlanner(q_lower, q_upper, collision_fn,
                                     step_size=0.25, seed=0)
    path = planner.plan(q_start, q_goal)      # list of configs or None
    path = shortcut_path(path, planner.edge_free, planner.rng)
"""

import numpy as np


class RRTConnectJointPlanner:
    """Bidirectional RRT-Connect in an N-dimensional joint space.

    Parameters
    ----------
    q_lower, q_upper : array-like (nj,)
        Joint limits (rad).  Samples and steer steps are confined to
        this box.
    collision_fn : callable (ndarray (nj,)) -> bool
        Returns True when the configuration collides.
    step_size : float
        Maximum L2 joint-space distance (rad) of a single tree
        extension.
    edge_resolution : float
        Maximum L2 joint-space distance (rad) between consecutive
        collision checks along an edge.
    max_iters : int
        Maximum sampling iterations before giving up.
    seed : int or None
        RNG seed for reproducible planning.
    project_fn : callable or None
        Optional ``project_fn(q_trial) -> q_projected | None``.  When given,
        each steered sample is projected onto a task-space constraint
        manifold before collision/edge checking.  This provides a lightweight
        constrained-RRT mode while preserving the historical unconstrained
        planner by default.
    """

    def __init__(self, q_lower, q_upper, collision_fn,
                 step_size=0.25, edge_resolution=0.06,
                 max_iters=3000, seed=0, project_fn=None,
                 max_project_step_factor=3.0):
        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        assert self.q_lower.shape == self.q_upper.shape
        assert np.all(self.q_upper > self.q_lower)
        self.nj            = self.q_lower.size
        self.collision_fn  = collision_fn
        self.step_size     = float(step_size)
        self.edge_res      = float(edge_resolution)
        self.max_iters     = int(max_iters)
        self.rng           = np.random.default_rng(seed)
        # Optional projection used for constrained planning.  The ordinary
        # planner is unchanged when ``project_fn`` is None.  When supplied,
        # every newly steered joint configuration is projected back onto the
        # requested task-space constraint manifold (e.g. fixed top-down EE
        # orientation) before edge validation.
        self.project_fn = project_fn
        self.max_project_step_factor = float(max_project_step_factor)

        # Diagnostics of the last plan() call
        self.last_iters   = 0
        self.last_n_nodes = 0
        self.last_failure = None   # None | "start" | "goal" | "iters"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, q_start, q_goal):
        """Plan a collision-free path from q_start to q_goal.

        Returns
        -------
        list[ndarray (nj,)] or None
            Sequence of configurations from start to goal (inclusive),
            or None when no path was found.  Consecutive configurations
            are guaranteed collision-free along the straight joint-space
            segment between them (at ``edge_resolution``).
        """
        q_start = np.asarray(q_start, dtype=float)
        q_goal  = np.asarray(q_goal, dtype=float)
        self.last_failure = None
        self.last_iters = 0

        if not self.within_limits(q_start) or self.collision_fn(q_start):
            self.last_failure = "start"
            return None
        if not self.within_limits(q_goal) or self.collision_fn(q_goal):
            self.last_failure = "goal"
            return None

        # Trivial case: direct connection
        if self.edge_free(q_start, q_goal):
            return [q_start.copy(), q_goal.copy()]

        # Two trees: nodes are (config, parent_index)
        tree_a = [(q_start.copy(), -1)]
        tree_b = [(q_goal.copy(), -1)]
        a_is_start = True

        for it in range(self.max_iters):
            self.last_iters = it + 1
            q_rand = self.rng.uniform(self.q_lower, self.q_upper)

            idx_new = self._extend(tree_a, q_rand)
            if idx_new is not None:
                q_new = tree_a[idx_new][0]
                idx_conn = self._connect(tree_b, q_new)
                if idx_conn is not None:
                    # Connected!  Assemble start -> goal path.
                    path_a = self._trace(tree_a, idx_new)   # root -> leaf
                    path_b = self._trace(tree_b, idx_conn)  # root -> leaf
                    if a_is_start:
                        path = path_a + path_b[::-1]
                    else:
                        path = path_b + path_a[::-1]
                    self.last_n_nodes = len(tree_a) + len(tree_b)
                    return self._dedup(path)

            # Swap trees (balanced growth)
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start

        self.last_failure = "iters"
        self.last_n_nodes = len(tree_a) + len(tree_b)
        return None

    def within_limits(self, q):
        return bool(np.all(q >= self.q_lower - 1e-9)
                    and np.all(q <= self.q_upper + 1e-9))

    def edge_free(self, qa, qb):
        """Check the straight joint-space segment qa -> qb for collisions."""
        qa = np.asarray(qa, dtype=float)
        qb = np.asarray(qb, dtype=float)
        dist = float(np.linalg.norm(qb - qa))
        n = max(1, int(math_ceil(dist / self.edge_res)))
        for i in range(1, n + 1):
            q = qa + (qb - qa) * (i / n)
            if self.collision_fn(q):
                return False
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _nearest(self, tree, q):
        pts = np.array([node[0] for node in tree])
        d = np.linalg.norm(pts - q, axis=1)
        return int(np.argmin(d))

    def _extend(self, tree, q_target):
        """One bounded step from the nearest node toward q_target.

        Returns the index of the new node, or None if blocked.
        """
        idx_near = self._nearest(tree, q_target)
        q_near = tree[idx_near][0]
        diff = q_target - q_near
        dist = float(np.linalg.norm(diff))
        if dist < 1e-12:
            return None
        reaches_valid_target = (
            self.project_fn is not None
            and dist <= self.step_size
            and self.within_limits(q_target)
            and not self.collision_fn(q_target)
        )
        if dist <= self.step_size:
            q_trial = q_target.copy()
        else:
            q_trial = q_near + diff * (self.step_size / dist)
        q_trial = np.clip(q_trial, self.q_lower, self.q_upper)

        q_new = q_trial
        if self.project_fn is not None and not reaches_valid_target:
            q_proj = self.project_fn(q_trial)
            if q_proj is None:
                return None
            q_new = np.asarray(q_proj, dtype=float)
            if q_new.shape != q_near.shape or not np.all(np.isfinite(q_new)):
                return None
            q_new = np.clip(q_new, self.q_lower, self.q_upper)
            # Reject pathological IK projections that jump to a remote branch
            # of the redundant manipulator.  Such jumps defeat the local RRT
            # extension assumption and usually create invalid interpolated
            # edges even when both endpoints satisfy the pose constraint.
            if (np.linalg.norm(q_new - q_near) >
                    self.max_project_step_factor * self.step_size):
                return None

        if np.linalg.norm(q_new - q_near) < 1e-10:
            return None
        if not self.edge_free(q_near, q_new):
            return None
        tree.append((q_new, idx_near))
        return len(tree) - 1

    def _connect(self, tree, q_target):
        """Repeat _extend toward q_target until reached or blocked.

        Returns the index of the node that reached q_target, or None.
        """
        while True:
            idx = self._extend(tree, q_target)
            if idx is None:
                return None
            if np.linalg.norm(tree[idx][0] - q_target) < 1e-9:
                return idx

    @staticmethod
    def _trace(tree, idx):
        """Path from tree root to node idx (root first)."""
        out = []
        while idx >= 0:
            q, parent = tree[idx]
            out.append(q.copy())
            idx = parent
        return out[::-1]

    @staticmethod
    def _dedup(path, tol=1e-10):
        out = [path[0]]
        for q in path[1:]:
            if np.linalg.norm(q - out[-1]) > tol:
                out.append(q)
        return out


def shortcut_path(path, edge_free_fn, rng, max_tries=80):
    """Random shortcutting: repeatedly try to replace sub-paths with a
    single straight segment when it is collision-free.

    Parameters
    ----------
    path : list[ndarray]
        Path from RRT-Connect (start and goal preserved).
    edge_free_fn : callable (qa, qb) -> bool
        Straight-segment validity check (same one the planner used).
    rng : numpy Generator
    max_tries : int
        Number of shortcut attempts.
    """
    if path is None or len(path) < 3:
        return path
    path = [q.copy() for q in path]
    for _ in range(max_tries):
        if len(path) < 3:
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))
        if edge_free_fn(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def math_ceil(x):
    n = int(x)
    return n if n == x else n + 1
