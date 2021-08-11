import scipy.interpolate

MPC_VELOCITY_MULTIPLIER = 0.5


class MPCAgent():

    def __init__(self, vec_env, controller):
        self.vec_env = vec_env
        self.controller = controller
        self.step_counter = 0

    def run(self, max_steps=10000):
        self.vec_env.reset()
        self.vec_env.get_state()
        while self.step_counter < max_steps:
            self.step_counter += 1
            current_time = self.step_counter / 120.0
            lin_speed, ang_speed = self._generate_example_linear_angular_speed(
                current_time)
            # lin_speed, ang_speed = (0., 0., 0.), 0.
            self._update_controller_params(lin_speed, ang_speed)
            self.controller.update()
            hybrid_action = self.controller.get_action()

            self.vec_env.step(hybrid_action)

    def _generate_example_linear_angular_speed(self, t):
        """Creates an example speed profile based on time for demo purpose."""
        vx = 0.6 * MPC_VELOCITY_MULTIPLIER
        vy = 0.2 * MPC_VELOCITY_MULTIPLIER
        wz = 0.8 * MPC_VELOCITY_MULTIPLIER

        time_points = (0, 5, 10, 15, 20, 25, 30)
        speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz), (0, -vy, 0, 0),
                        (0, 0, 0, 0), (0, 0, 0, wz))

        speed = scipy.interpolate.interp1d(
            time_points,
            speed_points,
            kind="previous",
            fill_value="extrapolate",
            axis=0)(
                t)

        return speed[0:3], speed[3]

    def _update_controller_params(self, lin_speed, ang_speed):
        self.controller.swing_leg_controller.set_desired_speed_as_tensor(lin_speed)
        self.controller.swing_leg_controller.set_desired_twisting_speed_as_tensor(ang_speed)
        self.controller.stance_leg_controller.set_desired_speed_as_tensor(lin_speed)
        self.controller.stance_leg_controller.set_desired_twisting_speed_as_tensor(ang_speed)
