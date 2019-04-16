#include <cmath>
#include <iostream>
#include <vector>

#include "Eigen/Dense"
#include "grader.h"

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;
using namespace Eigen;

vector<double> JMT(vector<double> &start, vector<double> &end, double T) {
  /**
   * Calculate the Jerk Minimizing Trajectory that connects the initial state
   * to the final state in time T.
   *
   * @param start - the vehicles start location given as a length three array
   *   corresponding to initial values of [s, s_dot, s_double_dot]
   * @param end - the desired end state for vehicle. Like "start" this is a
   *   length three array.
   * @param T - The duration, in seconds, over which this maneuver should occur.
   *
   * @output an array of length 6, each value corresponding to a coefficent in
   *   the polynomial:
   *   s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5
   *
   * EXAMPLE
   *   > JMT([0, 10, 0], [10, 10, 0], 1)
   *     [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
   */

  Matrix3f A;
  A << pow(T, 3), pow(T, 4), pow(T, 5),
      3 * pow(T, 2), 4 * pow(T, 3), 5 * pow(T, 4),
      6 * T, 12 * pow(T, 2), 20 * pow(T, 3);

  Vector3f b;
  double s_i = start[0];
  double s_i_dot = start[1];
  double s_i_dotdot = start[2];
  double s_f = end[0];
  double s_f_dot = end[1];
  double s_f_dotdot = end[2];

  b << s_f - (s_i + s_i_dot * T + 0.5 * s_i_dotdot * pow(T, 2)),
       s_f_dot - (s_i_dot + s_i_dotdot * T),
       s_f_dotdot - s_i_dotdot;

  Vector3f x = A.colPivHouseholderQr().solve(b);

  return {s_i, s_i_dot, 0.5 * s_i_dotdot, x[0], x[1], x[2]};
}

int main() {

  // create test cases
  vector<test_case> tc = create_tests();

  bool total_correct = true;

  for (int i = 0; i < tc.size(); ++i) {
    vector<double> jmt = JMT(tc[i].start, tc[i].end, tc[i].T);
    bool correct = close_enough(jmt, answers[i]);
    total_correct &= correct;
  }

  if (!total_correct) {
    std::cout << "Try again!" << std::endl;
  } else {
    std::cout << "Nice work!" << std::endl;
  }

  return 0;
}