import unittest
import itertools as its
from active_driver import *

class TestUniformRandomDriver(unittest.TestCase):

    def setUp(self):
        self.driver = UniformRandomDriver(10, 20)
        self.driver_seq = UniformRandomDriver(10, [20, 40, 60])
    
    def tearDown(self):
        pass

    def test_getinit(self):
        self.assertEqual(self.driver.num_classes, 10)
        self.assertEqual(self.driver.budgets_per_stage, 20)

    def test_get_plan(self):
        p = self.driver.get_plan()
        self.assertEqual(p, dict(zip(range(10), its.repeat(2))))
        self.driver.step()
        p = self.driver.get_plan()
        self.assertEqual(p, dict(zip(range(10), its.repeat(2))))

    def test_get_plan_seq(self):
        p = self.driver_seq.get_plan()
        self.assertEqual(p, dict(zip(range(10), its.repeat(2))))
        self.driver_seq.step()

        p = self.driver_seq.get_plan()
        self.assertEqual(p, dict(zip(range(10), its.repeat(4))))
        self.driver_seq.step()

        p = self.driver_seq.get_plan()
        self.assertEqual(p, dict(zip(range(10), its.repeat(6))))


class TestInverseAccuracyDriver(unittest.TestCase):

    def setUp(self):
        self.driver = InverseAccuracyDriver(5, 20)

    def tearDown(self):
        pass

    def test_get_plan(self):
        p = self.driver.get_plan()
        self.assertEqual(p, dict(zip(range(5), its.repeat(4))))
        fake_accuracy = [0.1, 0.2, 0.3, 0.4, 0.0]

        np.random.seed(17)
        self.driver.step(fake_accuracy)
        p = self.driver.get_plan()
        self.assertEqual(p, dict(zip(range(5), [8,5,2,1,4])))

