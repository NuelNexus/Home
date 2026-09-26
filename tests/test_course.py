import numpy as np

from drone_sim.course import Box, Cylinder, default_course, plan_smooth_path


def test_box_and_cylinder_sdf():
    b = Box((0, 0, 0), (2, 2, 2))
    np.testing.assert_allclose(b.sdf(np.array([[3.0, 0, 0], [0, 0, 0], [2, 2, 1]])), [2.0, -1.0, np.sqrt(2)])
    c = Cylinder(0, 0, 0.5, 2.0)
    np.testing.assert_allclose(c.sdf(np.array([[1.5, 0, 1.0], [0, 0, 3.0], [0, 0, 1.0]])), [1.0, 1.0, -0.5])


def test_planned_path_is_collision_free_and_reaches_goal():
    course = default_course()
    raw, key, path = plan_smooth_path(course, clearance=0.75)
    np.testing.assert_allclose(path[0], course.start, atol=1e-9)
    assert np.linalg.norm(path[-1] - np.asarray(course.goal)) < 0.06
    # every sample of the smooth path keeps the clearance (with the 10 % smoothing tolerance)
    assert course.sdf(path).min() > 0.75 * 0.9
    # the path really has to go through the windows: it crosses both walls inside their openings
    for gx, (cy, cz), (w, h) in [(9.5, (1.6, 2.8), (2.2, 2.0)), (15.5, (-1.8, 1.5), (2.2, 1.8))]:
        i = np.argmin(np.abs(path[:, 0] - gx))
        assert abs(path[i, 1] - cy) < w / 2 and abs(path[i, 2] - cz) < h / 2
