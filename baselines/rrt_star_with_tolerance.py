from random import random
from time import time
from pybullet_planning.motion_planners.rrt_star import OptimalNode, safe_path
from pybullet_planning.motion_planners.utils import INF, argmin, elapsed_time

EPSILON = 1e-6
PRINT_FREQUENCY = 100

def rrt_star_multi_goal(start, goal_fn, distance_fn, sample_fn, extend_fn, collision_fn, radius, is_goal_fn,
             max_time=INF, max_iterations=INF, goal_probability=.2, informed=False, verbose=False, draw_fn=None):
    """
    Modifications to RRT* to handle a variety of goals. Replace goal configuration with a goal sampling function.
    :param start: Start configuration - conf
    :param goal: Function to sample end configuration - goal_fn()->conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_time: Maximum runtime - float
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    if collision_fn(start):
        return None
    nodes = [OptimalNode(start)]
    goal_n = None
    start_time = time()
    iteration = 0
    while (elapsed_time(start_time) < max_time) and (iteration < max_iterations):
        do_goal = goal_n is None and (iteration == 0 or random() < goal_probability)
        s = goal_fn(config = True) if do_goal else sample_fn()
        if do_goal:
            goal = s
            if collision_fn(goal):
                continue


        # Informed RRT*
        if informed and (goal_n is not None) and (distance_fn(start, s) + distance_fn(s, goal) >= goal_n.cost):
            continue
        if iteration % PRINT_FREQUENCY == 0:
            success = goal_n is not None
            cost = goal_n.cost if success else INF
            if verbose:
                print('Iteration: {} | Time: {:.3f} | Success: {} | {} | Cost: {:.3f}'.format(
                    iteration, elapsed_time(start_time), success, do_goal, cost))
        iteration += 1

        nearest = argmin(lambda n: distance_fn(n.config, s), nodes)
        path = safe_path(extend_fn(nearest.config, s), collision_fn)
        if len(path) == 0:
            continue
        new = OptimalNode(path[-1], parent=nearest, d=distance_fn(
            nearest.config, path[-1]), path=path[:-1], iteration=iteration)
        # if safe and do_goal:
        #TODO: Replace to check if any node is within epsilon of goal
        if is_goal_fn(new.config):
            goal_n = new
            goal_n.set_solution(True)
        # if do_goal and (distance_fn(new.config, goal) < EPSILON):
        #     goal_n = new
        #     goal_n.set_solution(True)
        # TODO - k-nearest neighbor version
        neighbors = filter(lambda n: distance_fn(n.config, new.config) < radius, nodes)
        nodes.append(new)

        # TODO: smooth solution once found to improve the cost bound
        for n in neighbors:
            d = distance_fn(n.config, new.config)
            if (n.cost + d) < new.cost:
                path = safe_path(extend_fn(n.config, new.config), collision_fn)
                if (len(path) != 0) and (distance_fn(new.config, path[-1]) < EPSILON):
                    new.rewire(n, d, path[:-1], iteration=iteration)
        for n in neighbors:  # TODO - avoid repeating work
            d = distance_fn(new.config, n.config)
            if (new.cost + d) < n.cost:
                path = safe_path(extend_fn(new.config, n.config), collision_fn)
                if (len(path) != 0) and (distance_fn(n.config, path[-1]) < EPSILON):
                    n.rewire(new, d, path[:-1], iteration=iteration)

        if draw_fn:
            nodes[0].draw(draw_fn, remove_all=True)
            for n in nodes[1:]:
                n.draw(draw_fn, remove_all=False)

    if goal_n is None:
        return None
    return goal_n.retrace()

def informed_rrt_star_multi_goal(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, radius, **kwargs):
    return rrt_star_multi_goal(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, radius, informed=True, **kwargs)