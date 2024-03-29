#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

// // MPI support):
// #include <deal.II/lac/petsc_parallel_sparse_matrix.h>
// #include <deal.II/lac/petsc_parallel_vector.h>
// // for parallel computing
// #include <deal.II/lac/petsc_precondition.h>
// #include <deal.II/lac/petsc_solver.h>
// #include <deal.II/lac/petsc_sparse_matrix.h>
// #include <deal.II/lac/petsc_vector.h>
// #include <deal.II/distributed/shared_tria.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
// #include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// #include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "interpolation.h"

namespace LA {
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA

}  // namespace LA

using namespace dealii;

template <int dim>
class CoupledTH {
 public:
  CoupledTH(const unsigned int degree);
  ~CoupledTH();
  void run();

 private:
  void make_grid_and_dofs();
  void setup_P_system();
  void setup_T_system();
  void assemble_P_system();
  void assemble_T_system();
  void linear_solve_P();
  void linear_solve_T();
  void output_results(LA::MPI::Vector&, std::string) const;

  MPI_Comm mpi_communicator; //in mpi.h
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout; // in parallel program, each of MPI processes will write to the screen. This class solves this.
  TimerOutput computing_timer; //generate formatted output

  parallel::distributed::Triangulation<dim> triangulation;  // grid or meshes

  DoFHandler<dim> dof_handler;  // degree of freedom on vertices, lines, etc
  // DoFHandler<dim> dof_handler_couple;
  FE_Q<dim> fe;                 // Lagrange interpolation polynomials
  // FESystem<dim> fe_couple;
  // FE_Q<dim-2> fe_tube;
  QGauss<dim> quadrature_formula; //quadrature rules for numerical integration
  QGauss<dim - 1> face_quadrature_formula;
  // QGauss<dim - 2> line_quadrature_formula;

  IndexSet locally_owned_dofs; // subset of indices
  IndexSet locally_relevant_dofs;

  // AffineConstraints<double> T_constraints;

  const unsigned int degree;  // element degree

  // ConstraintMatrix constraints;  // hanging node
  LA::MPI::SparseMatrix P_system_matrix;  // M_P + K_P
  LA::MPI::SparseMatrix T_system_matrix;  // M_T + K_T
  LA::MPI::Vector P_system_rhs;           // right hand side of P
                                          // system
  LA::MPI::Vector T_system_rhs;           // right hand side of T
                                          // system

  LA::MPI::Vector P_locally_relevant_solution;  // P solution at n
  LA::MPI::Vector T_locally_relevant_solution;  // T solution at n

  LA::MPI::Vector old_P_locally_relevant_solution;  // P solution at n -1
  LA::MPI::Vector old_T_locally_relevant_solution;  // T solution at n -1

  Vector<double> initial_P_solution;  // P solution at 0
  Vector<double> initial_T_solution;  // T solution at 0

  double time;                   // t
  unsigned int timestep_number;  // n_t
  std::vector<double> time_sequence;

  double total_time;
  int n_time_step;
  double time_step;
  double period;

  unsigned int P_iteration_namber;
  unsigned int T_iteration_namber;
  unsigned int n_P_max_iteration = EquationData::n_g_P_max_iteration;
  unsigned int n_T_max_iteration = EquationData::n_g_T_max_iteration;
  double P_tol_residual = EquationData::g_P_tol_residual;
  double T_tol_residual = EquationData::g_T_tol_residual;
  Interpolation<3> data_interpolation;

  //PETScWrappers::MPI::SparseMatrix system_matrix;
  //PETScWrappers::MPI::Vector solution;
  //PETScWrappers::MPI::Vector system_rhs;
};

template <int dim>
CoupledTH<dim>::CoupledTH(const unsigned int degree)  // initialization
    : mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      triangulation(mpi_communicator),
      dof_handler(triangulation),
      // dof_handler_couple(triangulation),
      fe(degree),
      // fe_couple(FE_Q<dim>(degree),3),
      // fe_tube(degree),
      degree(degree),
      quadrature_formula(degree + 1),
      face_quadrature_formula(degree + 1),
      // line_quadrature_formula(degree + 1),
      time(0.0),
      timestep_number(0),
      period(EquationData::g_period),
      P_iteration_namber(0),
      T_iteration_namber(0),
      pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      computing_timer(mpi_communicator, pcout, TimerOutput::summary,
                      TimerOutput::wall_times), // Output wall clock times.
      data_interpolation(EquationData::dimension_x, EquationData::dimension_y,
                         EquationData::dimension_z,
                         EquationData::file_name_interpolation) {
  if (EquationData::is_linspace) {
    total_time = EquationData::g_total_time;
    n_time_step = EquationData::g_n_time_step;
    time_sequence = linspace(0.0, total_time, n_time_step);
    time_step = time_sequence[1] - time_sequence[0];
  } else {
    time_sequence = EquationData::g_time_sequence;
    n_time_step = time_sequence.size();
    total_time = time_sequence[n_time_step - 1];
    time_step = time_sequence[1] - time_sequence[0];
  }
}

template <int dim>
CoupledTH<dim>::~CoupledTH() {
  dof_handler.clear();
}

template <int dim>
void CoupledTH<dim>::make_grid_and_dofs() {

  GridIn<dim> gridin;  // instantiate a gridinput
  gridin.attach_triangulation(triangulation);
  std::ifstream f(EquationData::mesh_file_name);
  gridin.read_msh(f);
  // print_mesh_info(triangulation, "outputfiles/grid-1.eps");
  // triangulation.refine_global(1);
  dof_handler.distribute_dofs(fe);  // distribute dofs to grid globle
  // dof_handler_couple.distribute_dofs(fe_couple);
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  pcout << "Number of active cells: " << triangulation.n_active_cells() // the total number of active cells
        << " (on " << triangulation.n_levels() << " levels)" << std::endl // returns the level of the most refined active cell plus one (all cells with coarse, unrefined mesh)
        << "Number of degrees of freedom: " << 2 * dof_handler.n_dofs() << " ("
        << dof_handler.n_dofs() << '+' << dof_handler.n_dofs() << ')'
        << std::endl
        << std::endl; // n_dofs = number of shape functions that span this space

  initial_P_solution.reinit(dof_handler.n_dofs());
  initial_T_solution.reinit(dof_handler.n_dofs());
  old_P_locally_relevant_solution.reinit(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  old_T_locally_relevant_solution.reinit(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
}

template <int dim>
void CoupledTH<dim>::setup_P_system() {

  P_locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                     mpi_communicator);

  P_system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);

  // forming system matrixes and initialize these matrixesy
  P_system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                         mpi_communicator);
}

template <int dim>
void CoupledTH<dim>::setup_T_system() {

  T_locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                     mpi_communicator);

  T_system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);

  // forming system matrixes and initialize these matrixesy
  T_system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                         mpi_communicator);
}

template <int dim>
void CoupledTH<dim>::assemble_P_system() {
  cbgeo::Clock timer;
  timer.tick();

  // // reset matreix to zero  BECAUSE WE SETUP SYSTEM IN EACH TIME STEP SO IT
  // // IS NOT NECESSARY TO REINITALIZE IT.
  // P_system_matrix = 0;
  // P_system_rhs = 0;
  // P_locally_relevant_solution = 0;

  // Getting fe values
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  // define loop number
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  // store the value at previous step at q_point for T
  std::vector<double> old_T_sol_values(n_q_points);
  std::vector<Tensor<1, dim>> old_T_sol_grads(n_q_points);

  // store the value at previous step at q_point for P
  std::vector<double> old_P_sol_values(n_q_points);
  std::vector<Tensor<1, dim>> old_P_sol_grads(n_q_points);

  // store the rhs and bd and old solution value at q_point of element for P
  std::vector<double> P_source_values(n_q_points);
  std::vector<double> QP_bd_values(n_face_q_points);

  //  local element matrix
  FullMatrix<double> P_cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> P_cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> P_local_dof_indices(dofs_per_cell);

  // boudnary condition and source term
  EquationData::PressureSourceTerm<dim> P_source_term;
  EquationData::PressureNeumanBoundaryValues<dim> QP_boundary;
  EquationData::PressureDirichletBoundaryValues<dim> P_boundary;

  // loop for cell
  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  // double duration1 = 0.;
  // double duration2 = 0.;
  // double duration3 = 0.;
  // double duration4 = 0.;

  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {  // only assemble the system on cells that
                                     // acturally
                                     // belong to this MPI process

      // auto t1 = std::chrono::high_resolution_clock::now();

      // initialization
      P_cell_matrix = 0;
      P_cell_rhs = 0;
      fe_values.reinit(cell);

      // get the values at gauss point old solution from the system
      if (time < 1e-8) {
        // fe_values.get_function_values(initial_T_solution, old_T_sol_values);
        // fe_values.get_function_gradients(initial_T_solution, old_T_sol_grads);
        fe_values.get_function_values(initial_P_solution, old_P_sol_values);
        fe_values.get_function_gradients(initial_P_solution, old_P_sol_grads);
      } else {
        // fe_values.get_function_values(old_T_locally_relevant_solution,
        //                              old_T_sol_values);
        // fe_values.get_function_gradients(old_T_locally_relevant_solution,
        //                                  old_T_sol_grads);
        fe_values.get_function_values(old_P_locally_relevant_solution,
                                      old_P_sol_values);
        fe_values.get_function_gradients(old_P_locally_relevant_solution,
                                         old_P_sol_grads);
      }

      // get source term value at the gauss point
      P_source_term.set_time(time);
      P_source_term.value_list(fe_values.get_quadrature_points(),
                               P_source_values);  // 一列q个

      // loop for q_point ASSMBLING CELL METRIX (weak form equation writing)
      for (unsigned int q = 0; q < n_q_points; ++q) {

        const auto P_quadrature_coord = fe_values.quadrature_point(q);

        // 3d interp
        EquationData::g_perm = data_interpolation.value(P_quadrature_coord[0],
                                                        P_quadrature_coord[1],
                                                        P_quadrature_coord[2]);  // Functions::InterpolatedTensorProductGridData

        // pcout<<"\n"<<"at
        // point"<<"("<<P_quadrature_coord[0]<<","<<P_quadrature_coord[1]
        // <<","<<P_quadrature_coord[2]<<"), the permeability is
        // "<<EquationData::g_perm<<"\n";

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const Tensor<1, dim> grad_phi_i_P = fe_values.shape_grad(i, q);
          const double phi_i_P = fe_values.shape_value(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const Tensor<1, dim> grad_phi_j_P = fe_values.shape_grad(j, q);
            const double phi_j_P = fe_values.shape_value(j, q);
            // mass matrix
            P_cell_matrix(i, j) += (phi_i_P * phi_j_P * fe_values.JxW(q));
            // stiff matrix
            P_cell_matrix(i, j) +=
                (time_step * EquationData::g_perm * EquationData::g_B_w *
                 grad_phi_i_P * grad_phi_j_P * fe_values.JxW(q));
          }
          P_cell_rhs(i) += (time_step * phi_i_P * P_source_values[q] +
                            time_step * grad_phi_i_P * (Point<dim>(0, 0, 1)) *
                                (-EquationData::g_B_w * EquationData::g_perm *
                                 EquationData::g_P_grad) +
                            phi_i_P * old_P_sol_values[q]) *
                           fe_values.JxW(q);
        }
      }

      // auto t2 = std::chrono::high_resolution_clock::now();
      // duration1 +=
      //     std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
      //         .count();

      // auto tt1 = std::chrono::high_resolution_clock::now();

      // APPLIED NEWMAN BOUNDARY CONDITION
      for (unsigned int face_no = 0;
           face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
        if (cell->at_boundary(face_no)) {
          for (int bd_i = 0; bd_i < EquationData::g_num_QP_bnd_id; ++bd_i) {

            if (cell->face(face_no)->boundary_id() ==
                EquationData::g_QP_bnd_id[bd_i]) {
              fe_face_values.reinit(cell, face_no);

              // get boundary condition
              QP_boundary.get_bd_i(bd_i);
              QP_boundary.set_time(time);
              QP_boundary.set_boundary_id(*(EquationData::g_QP_bnd_id + bd_i));
              QP_boundary.value_list(fe_face_values.get_quadrature_points(),
                                     QP_bd_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q) {

                const auto P_face_quadrature_coord =
                    fe_face_values.quadrature_point(q);

                // 3d interp
                EquationData::g_perm = data_interpolation.value(
                    P_face_quadrature_coord[0], P_face_quadrature_coord[1],
                    P_face_quadrature_coord[2]);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  P_cell_rhs(i) += -time_step * EquationData::g_B_w *
                                   (fe_face_values.shape_value(i, q) *
                                    QP_bd_values[q] * fe_face_values.JxW(q));
                }
              }
            }
          }
        }
      }

      // auto tt2 = std::chrono::high_resolution_clock::now();
      // duration2 +=
      //     std::chrono::duration_cast<std::chrono::microseconds>(tt2 - tt1)
      //         .count();

      // local ->globe
      cell->get_dof_indices(P_local_dof_indices);

      // auto ttt1 = std::chrono::high_resolution_clock::now();

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          P_system_matrix.add(P_local_dof_indices[i], P_local_dof_indices[j],
                              P_cell_matrix(i, j));  // sys_mass matrix
        }
        P_system_rhs(P_local_dof_indices[i]) += P_cell_rhs(i);
      }

      // auto ttt2 = std::chrono::high_resolution_clock::now();
      // duration3 +=
      //     std::chrono::duration_cast<std::chrono::microseconds>(ttt2 - ttt1)
      //         .count();
    }
  }

  P_system_matrix.compress(VectorOperation::add);
  P_system_rhs.compress(VectorOperation::add);

  // auto tttt1 = std::chrono::high_resolution_clock::now();
  // ADD DIRICHLET BOUNDARY
  {

    for (unsigned int bd_i = 0; bd_i < EquationData::g_num_P_bnd_id; ++bd_i) {

      P_boundary.get_bd_i(bd_i);
      P_boundary.set_time(time);
      P_boundary.set_boundary_id(*(EquationData::g_P_bnd_id + bd_i));
      std::map<types::global_dof_index, double> P_bd_values;
      VectorTools::interpolate_boundary_values(
          dof_handler, *(EquationData::g_P_bnd_id + bd_i), P_boundary,
          P_bd_values);  // i is boundary index
      LA::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
      MatrixTools::apply_boundary_values(P_bd_values, P_system_matrix, tmp,
                                         P_system_rhs, false);
      P_locally_relevant_solution = tmp;
    }
  }

  // auto tttt2 = std::chrono::high_resolution_clock::now();

  // duration4 +=
  //     std::chrono::duration_cast<std::chrono::microseconds>(tttt2 - tttt1)
  //         .count();

  // std::cout << "\n"
  //           << "computing cell matrix and rhs"
  //           << ": " << duration1 / 1000 << " ms";
  // std::cout << "\n"
  //           << "apply neuman boundary "
  //           << ": " << duration2 / 1000 << " ms";
  // std::cout << "\n"
  //           << "assembling system matrix"
  //           << ": " << duration3 / 1000 << " ms";
  // std::cout << "\n"
  //           << "apply dirichlet boundary"
  //           << ": " << duration4 / 1000 << " ms";

  timer.tock("assemble_P_system");
}

template <int dim>
void CoupledTH<dim>::assemble_T_system() {
  cbgeo::Clock timer;
  timer.tick();
  // // reset matreix to zero NOT NECESSARY
  // T_system_matrix = 0;
  // T_system_rhs = 0;
  // T_locally_relevant_solution = 0;

  // Getting fe values
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  // define loop number
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  // store the value at previous step at q_point for T
  std::vector<double> old_T_sol_values(n_q_points);
  std::vector<Tensor<1, dim>> old_T_sol_grads(n_q_points);

  // store the value at previous step at q_point for P
  std::vector<double> old_P_sol_values(n_q_points);
  std::vector<Tensor<1, dim>> old_P_sol_grads(n_q_points);

  // store the source and bd value at q_point
  std::vector<double> T_source_values(n_q_points);
  std::vector<double> QT_bd_values(n_face_q_points);

  //  local element matrix
  FullMatrix<double> T_cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> T_cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> T_local_dof_indices(dofs_per_cell);

  // boudnary condition
  EquationData::TemperatureSourceTerm<dim> T_source_term;
  EquationData::TemperatureNeumanBoundaryValues<dim> QT_boundary;
  EquationData::TemperatureDirichletBoundaryValues<dim> T_boundary;

  // loop for cell
  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {
      // initialization
      T_cell_matrix = 0;
      T_cell_rhs = 0;
      fe_values.reinit(cell);

      // get the values at gauss point old solution from the system
      if (time < 1e-8) {
        fe_values.get_function_values(initial_T_solution, old_T_sol_values);
        fe_values.get_function_gradients(initial_T_solution, old_T_sol_grads);
        fe_values.get_function_values(initial_P_solution, old_P_sol_values);
        fe_values.get_function_gradients(initial_P_solution, old_P_sol_grads);
      } else {
        fe_values.get_function_values(old_T_locally_relevant_solution,
                                      old_T_sol_values);
        fe_values.get_function_gradients(old_T_locally_relevant_solution,
                                         old_T_sol_grads);
        fe_values.get_function_values(old_P_locally_relevant_solution,
                                      old_P_sol_values);
        fe_values.get_function_gradients(old_P_locally_relevant_solution,
                                         old_P_sol_grads);
      }

      // get right hand side
      T_source_term.set_time(time);
      T_source_term.value_list(fe_values.get_quadrature_points(),
                               T_source_values);

      // loop for q_point ASSMBLING CELL METRIX (weak form equation writing)
      for (unsigned int q = 0; q < n_q_points; ++q) {
        const auto T_quadrature_coord = fe_values.quadrature_point(q);

        // 3d interp
        EquationData::g_perm = data_interpolation.value(T_quadrature_coord[0],
                                                        T_quadrature_coord[1],
                                                        T_quadrature_coord[2]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const Tensor<1, dim> grad_phi_i_T = fe_values.shape_grad(i, q);
          const double phi_i_T = fe_values.shape_value(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const Tensor<1, dim> grad_phi_j_T = fe_values.shape_grad(j, q);
            const double phi_j_T = fe_values.shape_value(j, q);
            // mass matrix
            T_cell_matrix(i, j) += (phi_i_T * phi_j_T * fe_values.JxW(q));
            // stiff matrix
            T_cell_matrix(i, j) +=
                time_step * (EquationData::g_lam / EquationData::g_c_T *
                             grad_phi_i_T * grad_phi_j_T * fe_values.JxW(q));
            // conv matrix
            T_cell_matrix(i, j) +=
                time_step * EquationData::g_c_w / EquationData::g_c_T *
                phi_i_T *
                (-EquationData::g_perm *
                 (old_P_sol_grads[q] +
                  (Point<dim>(0, 0, 1)) * EquationData::g_P_grad) *
                 grad_phi_j_T * fe_values.JxW(q));
          }

          T_cell_rhs(i) +=
              (time_step * T_source_values[q] + old_T_sol_values[q]) * phi_i_T *
              fe_values.JxW(q);
        }
      }

      // APPLIED NEUMAN BOUNDARY CONDITION

      for (unsigned int face_no = 0;
           face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
        if (cell->at_boundary(face_no)) {
          for (int bd_i = 0; bd_i < EquationData::g_num_QT_bnd_id; ++bd_i) {
            if (cell->face(face_no)->boundary_id() ==
                EquationData::g_QT_bnd_id[bd_i]) {
              fe_face_values.reinit(cell, face_no);

              // get boundary condition
              QT_boundary.get_bd_i(bd_i);
              QT_boundary.set_time(time);
              QT_boundary.set_boundary_id(*(EquationData::g_QT_bnd_id + bd_i));
              QT_boundary.value_list(fe_face_values.get_quadrature_points(),
                                     QT_bd_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q) {

                const auto T_face_quadrature_coord =
                    fe_face_values.quadrature_point(q);

                // 3d interp
                EquationData::g_perm = data_interpolation.value(
                    T_face_quadrature_coord[0], T_face_quadrature_coord[1],
                    T_face_quadrature_coord[2]);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  T_cell_rhs(i) += -time_step / EquationData::g_c_T *
                                   fe_face_values.shape_value(i, q) *
                                   QT_bd_values[q] * fe_face_values.JxW(q);
                }
              }
            }
          }
        }
      }
      // local ->globe
      cell->get_dof_indices(T_local_dof_indices);
      // T_constraints.distribute_local_to_global(T_cell_matrix, T_cell_rhs,
      //                                          T_local_dof_indices,
      //                                          T_system_matrix,
      //                                          T_system_rhs);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          T_system_matrix.add(T_local_dof_indices[i], T_local_dof_indices[j],
                              T_cell_matrix(i, j));  //
        }
        T_system_rhs(T_local_dof_indices[i]) += T_cell_rhs(i);
      }
    }
  }

  // compress the matrix
  T_system_matrix.compress(VectorOperation::add);
  T_system_rhs.compress(VectorOperation::add);

  // ADD DIRICHLET BOUNDARY
  {

    for (int bd_i = 0; bd_i < EquationData::g_num_T_bnd_id; bd_i++) {
      T_boundary.get_bd_i(bd_i);
      T_boundary.get_period(period);  // for seeting sin function
      T_boundary.set_time(time);
      T_boundary.set_boundary_id(*(EquationData::g_T_bnd_id + bd_i));
      std::map<types::global_dof_index, double> T_bd_values;

      VectorTools::interpolate_boundary_values(
          dof_handler, *(EquationData::g_T_bnd_id + bd_i), T_boundary,
          T_bd_values);  // i is boundary index

      LA::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
      MatrixTools::apply_boundary_values(T_bd_values, T_system_matrix, tmp,
                                         T_system_rhs, false);
      T_locally_relevant_solution = tmp;
    }
  }

  timer.tock("assemble_T_system");
}

template <int dim>
void CoupledTH<dim>::linear_solve_P() {
  cbgeo::Clock timer;
  timer.tick();

  LA::MPI::Vector distributed_P_solution(locally_owned_dofs, mpi_communicator);
  SolverControl solver_control(
      std::max<std::size_t>(n_P_max_iteration, P_system_rhs.size()),
      P_tol_residual * P_system_rhs.l2_norm());  // setting for solver
  // pcout<< "\n the l1 norm of the P_system is"<< P_system_matrix.l1_norm() <<
  // "\n";
  distributed_P_solution = P_locally_relevant_solution;

  // LA::SolverGMRES solver(solver_control, mpi_communicator);  // config cg
  LA::SolverCG solver(solver_control, mpi_communicator);  // config cg
  // LA::MPI::PreconditionJacobi preconditioner(P_system_matrix);
  PETScWrappers::PreconditionBlockJacobi preconditioner(P_system_matrix);
  solver.solve(P_system_matrix, distributed_P_solution, P_system_rhs,
               preconditioner);  // solve eq

  P_locally_relevant_solution = distributed_P_solution;

  old_P_locally_relevant_solution = distributed_P_solution;

  P_iteration_namber = solver_control.last_step();

  timer.tock("linear_solve_P");
  pcout << "\nIterations required for P convergence: " << P_iteration_namber
        << "\n";
}

template <int dim>
void CoupledTH<dim>::linear_solve_T() {
  cbgeo::Clock timer;
  timer.tick();
  LA::MPI::Vector distributed_T_solution(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(
      std::max<std::size_t>(n_T_max_iteration, T_system_rhs.size()),
      T_tol_residual * T_system_rhs.l2_norm());  // setting for solver
  // pcout<< "\n the l1 norm of the T_system is"<< T_system_matrix.l1_norm() <<
  // "\n";

  distributed_T_solution = T_locally_relevant_solution;
  LA::SolverGMRES solver(solver_control,
                         mpi_communicator);  // config solver
  // LA::MPI::PreconditionAMG preconditioner;
  // LA::MPI::PreconditionAMG::AdditionalData data;
  // data.symmetric_operator = false;
  // preconditioner.initialize(T_system_matrix, data);

  LA::MPI::PreconditionJacobi preconditioner(T_system_matrix);  // precond
  // preconditioner.initialize(T_system_matrix, 1.0);      // initialize precond
  solver.solve(T_system_matrix, distributed_T_solution, T_system_rhs,
               preconditioner);  // solve eq

  // T_constraints.distribute(distributed_T_solution);
  T_locally_relevant_solution = distributed_T_solution;

  old_T_locally_relevant_solution = distributed_T_solution;

  T_iteration_namber = solver_control.last_step();

  timer.tock("linear_solve_T");

  pcout << " \nIterations required for T convergence:    " << T_iteration_namber
        << "\n";
}

// @sect4{<code>CoupledTH::output_results</code>}
//
// Neither is there anything new in generating graphical output:
template <int dim>
void CoupledTH<dim>::output_results(LA::MPI::Vector& solution,
                                    std::string var_name) const {

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(solution, var_name);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =
      var_name + "-solution-" + Utilities::int_to_string(timestep_number, 3);

  const std::string pvtu_master_filename = data_out.write_vtu_with_pvtu_record(
      "outputfiles/", filename, timestep_number, mpi_communicator);

  // In a parallel setting, several files are typically written per time step.
  // The number of files written in parallel depends on the number of MPI
  // processes (see parameter mpi_communicator), and a specified number of
  // n_groups with default value 0. The background is that VTU file output
  // supports grouping files from several CPUs into a given number of files
  // using MPI I/O when writing on a parallel filesystem. The default value of
  // n_groups is 0, meaning that every MPI rank will write one file. A value of
  // 1 will generate one big file containing the solution over the whole domain,
  // while a larger value will create n_groups files (but not more than there
  // are MPI ranks).

  if (this_mpi_process == 0) {
    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back(
        std::pair<double, std::string>(time, pvtu_master_filename));
    std::ofstream pvd_output("outputfiles/" + var_name + "_solution.pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }
}

template <int dim>
void CoupledTH<dim>::run() {
  cbgeo::Clock timer;
  timer.tick();

  unsigned int binary_search_number;
  double initial_time_step;
  double theta;

  make_grid_and_dofs();

  // setup_P_system();
  // setup_T_system();

  VectorTools::interpolate(dof_handler,
                           EquationData::TemperatureInitialValues<dim>(),
                           initial_T_solution);
  VectorTools::interpolate(dof_handler,
                           EquationData::PressureInitialValues<dim>(),
                           initial_P_solution);

  do {

    pcout << "\nTimestep " << timestep_number;

    binary_search_number = 1;
    initial_time_step =
        time_sequence[timestep_number + 1] - time_sequence[timestep_number];
    time_step = initial_time_step / 2;
    theta = 0;

    do {

      setup_P_system();

      setup_T_system();

      assemble_P_system();

      linear_solve_P();

      assemble_T_system();

      linear_solve_T();

      time += time_step;

      theta += pow(0.5, binary_search_number);

      if (P_iteration_namber > n_P_max_iteration / 2 ||
          T_iteration_namber > n_T_max_iteration / 2) {
        time_step = time_step / 2;
        ++binary_search_number;
      }

      pcout << "   \n theta  = " << theta << std::endl;

    } while ((1 - theta) > 0.00001);

    pcout << "   \n Solver converged in " << binary_search_number
          << " time divisions." << std::endl;

    timestep_number += 1;

    pcout << "\nt=" << time << ", dt=" << time_step << '.' << std::endl;

    output_results(T_locally_relevant_solution, "T");
    output_results(P_locally_relevant_solution, "P");

    pcout << "\n" << std::endl << std::endl;

    // MatrixOut matrix_out;
    // std::ofstream out_T_matrix
    // ("/outputfiles/2rhs_T_matrix_at_"+std::to_string(time));
    // matrix_out.build_patches (T_system_matrix, "T_system_matrix");
    // matrix_out.write_gnuplot (out_T_matrix);
    // T_system_matrix.print_formatted(out_T_matrix);
    // T_system_rhs.print(out);

  } while (time < total_time);

  timer.tock("solve_all");
  pcout << "\n" << std::endl << std::endl;
}
